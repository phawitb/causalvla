(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ResultDashboard = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function filterModelsForResultView(models, view) {
    if (view !== 'm-models') return models;
    return models.filter(model => /^M\d+(?:-|$)/.test(model.id));
  }

  function defaultModelForResultView(models, view) {
    return filterModelsForResultView(models, view)[0]?.id ?? null;
  }

  function resultsManifestRequestOptions() {
    return {cache: 'no-store'};
  }

  function resultCollectionForView(data, view) {
    if (view === 'm-models-fixed') {
      return {models: data.fixedModels || [], runs: data.fixedRuns || [], episodes: data.fixedEpisodes || []};
    }
    return {models: data.models || [], runs: data.runs || [], episodes: data.episodes || []};
  }

  function seedResultTables(models, runs) {
    const modelIds = new Set(models.map(model => model.id));
    const visibleRuns = runs.filter(run => modelIds.has(run.model));
    const seeds = [...new Set(visibleRuns.map(run => Number(run.seed)))].sort((a, b) => a - b);
    return seeds.map(seed => ({
      seed,
      rows: models.map(model => {
        const matching = visibleRuns.filter(run => run.model === model.id && Number(run.seed) === seed);
        const successes = matching.reduce((sum, run) => sum + Number(run.successes || 0), 0);
        const episodes = matching.reduce((sum, run) => sum + Number(run.episodes || 0), 0);
        const levels = {};
        matching.forEach(run => {
          const level = Number(run.level);
          const current = levels[level] || {successes: 0, episodes: 0};
          current.successes += Number(run.successes || 0);
          current.episodes += Number(run.episodes || 0);
          levels[level] = current;
        });
        Object.keys(levels).forEach(level => {
          const value = levels[level];
          levels[level] = value.episodes ? value.successes * 100 / value.episodes : null;
        });
        return {
          id: model.id,
          name: model.name,
          successes,
          episodes,
          overall: episodes ? successes * 100 / episodes : null,
          levels,
        };
      }),
    }));
  }

  return {filterModelsForResultView, defaultModelForResultView, resultsManifestRequestOptions, resultCollectionForView, seedResultTables};
}));
