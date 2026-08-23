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

  return {filterModelsForResultView, defaultModelForResultView, resultsManifestRequestOptions, resultCollectionForView};
}));
