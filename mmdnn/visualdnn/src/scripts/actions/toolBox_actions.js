export const addLayer = (layer) => {
  return {
    type: 'ADD_LAYER',
    layerName: layer,
  }
}

export const importModel = (json) => {
  return {
    type: 'IMPORT_MODEL',
    model: json
  }
}

export const addKerasDataset = (name) => {
  return {
    type: 'ADD_KERAS_DATASET',
    name
  }
}

export const addCompile = (newCompile) => {
  return {
    type: 'ADD_COMPILE',
    compile: newCompile
  }
}

export const addRun = (newRun) => {
  return {
    type: 'ADD_RUN',
    run: newRun
  }
}