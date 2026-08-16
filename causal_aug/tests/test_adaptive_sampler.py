import pytest
import torch

from causal_aug import INTERVENTION_FAMILIES, PacerContextualBandit


def test_context_assignment_is_balanced_by_rank():
    bandit = PacerContextualBandit(warmup_steps=0)

    contexts = bandit.assign_context(torch.tensor([9.0, 1.0, 5.0, 2.0, 8.0, 4.0]))

    assert contexts.tolist() == [2, 0, 1, 0, 2, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"exploration_floor": -0.1},
        {"exploration_floor": 1.1},
        {"ema_decay": -0.1},
        {"ema_decay": 1.0},
        {"warmup_steps": -1},
    ],
)
def test_bandit_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PacerContextualBandit(**kwargs)


@pytest.mark.parametrize("losses", [torch.tensor([]), torch.ones(2, 2), torch.tensor([float("nan")])])
def test_context_assignment_rejects_invalid_losses(losses):
    with pytest.raises(ValueError):
        PacerContextualBandit().assign_context(losses)


def test_warmup_probabilities_are_uniform():
    bandit = PacerContextualBandit(exploration_floor=0.2, warmup_steps=10)

    probabilities = bandit.probabilities(torch.tensor([0, 1, 2]))

    assert torch.allclose(probabilities, torch.full_like(probabilities, 1 / probabilities.shape[1]))


def test_exploration_floor_bounds_every_arm():
    bandit = PacerContextualBandit(exploration_floor=0.2, warmup_steps=0)
    bandit.reward_ema[0, 0] = 100.0
    bandit.counts[0, :] = 1

    probabilities = bandit.probabilities(torch.tensor([0]))[0]

    assert probabilities.sum().item() == pytest.approx(1.0)
    assert torch.all(probabilities >= 0.2 / probabilities.numel())


def test_unobserved_context_remains_uniform_after_warmup():
    bandit = PacerContextualBandit(warmup_steps=0)
    bandit.reward_ema[0, 0] = 10.0
    bandit.counts[0, 0] = 1

    probabilities = bandit.probabilities(torch.tensor([2]))[0]

    assert torch.allclose(probabilities, torch.full_like(probabilities, 1 / probabilities.numel()))


def test_update_is_context_and_arm_specific():
    bandit = PacerContextualBandit(ema_decay=0.5, warmup_steps=0)

    rejected = bandit.update(
        torch.tensor([0, 0, 2]),
        torch.tensor([1, 1, 3]),
        torch.tensor([2.0, 4.0, 8.0]),
    )

    assert rejected.item() == 0
    assert bandit.reward_ema[0, 1].item() == pytest.approx(3.0)
    assert bandit.reward_ema[2, 3].item() == pytest.approx(8.0)
    assert bandit.counts[0, 1].item() == 2
    assert bandit.steps.item() == 1


def test_existing_reward_uses_ema_update():
    bandit = PacerContextualBandit(ema_decay=0.5, warmup_steps=0)
    bandit.update(torch.tensor([1]), torch.tensor([2]), torch.tensor([2.0]))

    bandit.update(torch.tensor([1]), torch.tensor([2]), torch.tensor([6.0]))

    assert bandit.reward_ema[1, 2].item() == pytest.approx(4.0)
    assert bandit.counts[1, 2].item() == 2


def test_nonfinite_reward_is_rejected():
    bandit = PacerContextualBandit(warmup_steps=0)

    rejected = bandit.update(
        torch.tensor([1]), torch.tensor([2]), torch.tensor([float("nan")])
    )

    assert rejected.item() == 1
    assert bandit.counts.sum().item() == 0
    assert bandit.steps.item() == 1


def test_state_dict_round_trip_preserves_adaptation():
    source = PacerContextualBandit(warmup_steps=0)
    source.update(torch.tensor([2]), torch.tensor([4]), torch.tensor([3.0]))
    target = PacerContextualBandit(warmup_steps=0)

    target.load_state_dict(source.state_dict())

    assert torch.equal(target.reward_ema, source.reward_ema)
    assert torch.equal(target.counts, source.counts)
    assert torch.equal(target.steps, source.steps)


def test_application_preserves_camera_shapes_range_and_sources():
    torch.manual_seed(17)
    bandit = PacerContextualBandit(warmup_steps=0)
    images = [torch.zeros(4, 3, 32, 32), torch.zeros(4, 3, 32, 32)]
    originals = [image.clone() for image in images]
    choices = torch.tensor([0, 0, 4, 4])

    outputs = bandit.apply(images, choices, intensity=0.5)

    assert len(outputs) == 2
    for source, original, output in zip(images, originals, outputs, strict=True):
        assert torch.equal(source, original)
        assert output.shape == source.shape
        assert torch.isfinite(output).all()
        assert output.min() >= -1 and output.max() <= 1
    assert not torch.equal(outputs[0], images[0])
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    ("images", "choices", "intensity"),
    [
        ([], torch.tensor([0]), 0.5),
        ([torch.zeros(2, 3, 8, 8), torch.zeros(3, 3, 8, 8)], torch.tensor([0, 1]), 0.5),
        ([torch.zeros(2, 3, 8, 8)], torch.tensor([0]), 0.5),
        ([torch.zeros(2, 3, 8, 8)], torch.tensor([0, len(INTERVENTION_FAMILIES)]), 0.5),
        ([torch.zeros(2, 3, 8, 8)], torch.tensor([0, 1]), -0.1),
    ],
)
def test_application_rejects_invalid_inputs(images, choices, intensity):
    with pytest.raises(ValueError):
        PacerContextualBandit().apply(images, choices, intensity)
