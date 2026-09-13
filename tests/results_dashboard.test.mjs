import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';

const require = createRequire(import.meta.url);
const html = readFileSync(new URL('../pipeline.html', import.meta.url), 'utf8');
const {
  defaultModelForResultView,
  filterModelsForResultView,
  resultsManifestRequestOptions,
  fetchResultsManifest,
  resultCollectionForView,
  seedResultTables,
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

test('fetchResultsManifest reloads dashboard data without browser cache', async () => {
  const calls = [];
  const payload = {fixedSuites: {libero_object: {models: []}}};
  const result = await fetchResultsManifest(async (url, options) => {
    calls.push([url, options]);
    return {ok: true, json: async () => payload};
  });

  assert.deepEqual(calls, [['results-data.json', {cache: 'no-store'}]]);
  assert.equal(result, payload);
});

test('M-Models Fix selects only the fixed result collection', () => {
  const data = {models, runs: ['original'], episodes: ['original'], fixedModels: [{id: 'M0-clean'}], fixedRuns: ['fixed'], fixedEpisodes: ['fixed']};
  assert.deepEqual(resultCollectionForView(data, 'm-models-fixed'), {models: data.fixedModels, runs: data.fixedRuns, episodes: data.fixedEpisodes});
  assert.equal(resultCollectionForView(data, 'all').runs, data.runs);
});

test('M-Models Fix selects the requested LIBERO suite', () => {
  const spatial = {models: [{id: 'M0-clean'}], runs: ['spatial'], episodes: []};
  const object = {models: [{id: 'M5-v2-warm-070'}], runs: ['object'], episodes: []};
  const data = {fixedSuites: {libero_spatial: spatial, libero_object: object}};

  assert.deepEqual(resultCollectionForView(data, 'm-models-fixed', 'libero_object'), object);
  assert.deepEqual(resultCollectionForView(data, 'm-models-fixed', 'libero_spatial'), spatial);
});

test('fixed result label lists every evaluation seed', () => {
  assert.match(html, /Fixed per episode · Seeds 4000, 5000, 6000/);
});

test('fixed results expose spatial and object suite choices', () => {
  assert.match(html, /id="results-suite-filter"/);
  assert.match(html, /value="libero_spatial"/);
  assert.match(html, /value="libero_object"/);
});

test('changing fixed suite reloads the results manifest', () => {
  assert.match(html, /results-suite-filter'[\s\S]*?await reloadResultsData\(\)/);
});

test('dashboard script URL is cache-versioned', () => {
  assert.match(html, /scripts\/results_dashboard\.js\?v=20260913-object-refresh/);
});

test('seed result tables aggregate runs by model and level while preserving missing cells', () => {
  const visibleModels = [{id: 'M0-clean', name: 'M0'}, {id: 'M1-offline-dr', name: 'M1'}];
  const runs = [
    {model: 'legacy', seed: 1000, level: 0, successes: 9, episodes: 10},
    {model: 'M0-clean', seed: 4000, level: 0, successes: 6, episodes: 10},
    {model: 'M0-clean', seed: 4000, level: 1, successes: 4, episodes: 10},
    {model: 'M1-offline-dr', seed: 5000, level: 2, successes: 7, episodes: 10},
  ];

  assert.deepEqual(seedResultTables(visibleModels, runs), [
    {seed: 4000, rows: [
      {id: 'M0-clean', name: 'M0', successes: 10, episodes: 20, overall: 50, levels: {0: 60, 1: 40}},
      {id: 'M1-offline-dr', name: 'M1', successes: 0, episodes: 0, overall: null, levels: {}},
    ]},
    {seed: 5000, rows: [
      {id: 'M0-clean', name: 'M0', successes: 0, episodes: 0, overall: null, levels: {}},
      {id: 'M1-offline-dr', name: 'M1', successes: 7, episodes: 10, overall: 70, levels: {2: 70}},
    ]},
  ]);
});
