let code = `import keras
from keras.models import Sequential
model = Sequential()
`
let index = 0
const initialState = {
  layers: [],
  dataset: '',
  compile: {},
  run: {},
  selectedLayer: undefined,//string, the id of the selected layer
}
export default initialState