const fs = require('fs')
const { spawn, exec } = require('child_process')

export {
  delLink,
  addLink,
  selectLayer,
  copyLayers,
  delLayers,
  folded,
  buildModule,
  shareModule
} from './graph_actions'

export {
  changePar,
  fetchShapes,
  getShapes
} from 'par_actions'

export {
  addLayer, 
  importModel, 
  addKerasDataset,
  addCompile,
  addRun
} from 'toolBox_actions'

export {
  changeCode
} from 'code_actions'

// const node_ssh = require('node-ssh')
// let ssh = new node_ssh()
// let connection = ssh.connect({
//   host: '10.150.146.105',
//   username: 'v-qianww',
//   password: '123',
// })


// const venv = `python3`




// export const moveLayers = (from, to) => {
//   return {
//     type: 'MOVE_LAYERS',
//     from,
//     to,
//   }
// }





// async actions

