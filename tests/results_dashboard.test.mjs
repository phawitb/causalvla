import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);
const {
  defaultModelForResultView,
  filterModelsForResultView,
  resultsManifestRequestOptions,
  resultCollectionForView,
} = require('../scripts/results_dashboard.js');

const models = [
  {id: 'a', name: 'Model A'},
  {id: 'M0-clean', name: 'M0 — Clean'},
  {id: 'M1-offline-dr', name: 'M1 — Offline DR'},
  {id: 'M2-online-dr', name: 'M2 — Online DR'},
  {id: 'M3-v2-warm', name: 'M3 — V2-Warm'},
  {id: 'v2_warm', name: 'V2-Warm'},
];

test('M-Models view exposes only numbered fair-protocol models', () => {
  assert.deepEqual(
    filterModelsForResultView(models, 'm-models').map(model => model.id),
    ['M0-clean', 'M1-offline-dr', 'M2-online-dr', 'M3-v2-warm'],
  );
});

test('All Models view preserves every result model', () => {
  assert.deepEqual(
    filterModelsForResultView(models, 'all').map(model => model.id),
    models.map(model => model.id),
  );
});

test('switching result views selects the first visible model', () => {
  assert.equal(defaultModelForResultView(models, 'm-models'), 'M0-clean');
  assert.equal(defaultModelForResultView([{id: 'a'}], 'm-models'), null);
});

test('results manifest requests bypass stale browser caches', () => {
  assert.deepEqual(resultsManifestRequestOptions(), {cache: 'no-store'});
});

test('M-Models Fix selects only the fixed result collection', () => {
  const data = {models, runs: ['original'], episodes: ['original'], fixedModels: [{id: 'M0-clean'}], fixedRuns: ['fixed'], fixedEpisodes: ['fixed']};
  assert.deepEqual(resultCollectionForView(data, 'm-models-fixed'), {models: data.fixedModels, runs: data.fixedRuns, episodes: data.fixedEpisodes});
  assert.equal(resultCollectionForView(data, 'all').runs, data.runs);
});
