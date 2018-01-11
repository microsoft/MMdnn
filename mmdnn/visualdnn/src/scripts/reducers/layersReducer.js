
import { getLayer, getID } from '../../assets/def';
import { findInLayers, findOutLayers } from "../ulti/ulti_code"
import { normalize, denormalize } from 'normalizr'
import {
  deepCopy,
  search
} from "../ulti/ulti_layers"
import {
  layerSchema,
  layersSchema
} from '../ulti/schema'
import update from 'react-addons-update'

let layerIndex = 0;
let moduleIndex = 0;
export const config = (state = { result: [], entities: {layers:{},attrs:{}} }, action) => {
  let oldConfig = { ...state }
  let oldLayers = denormalize(oldConfig.result, layersSchema, oldConfig.entities)
  switch (action.type) {
    case "CHANGE_PAR":
      let { attr } = action
      let name = attr.pars.name, id = attr.id
      return update(oldConfig, {
        entities: {
          attrs: {
            [action.attr.id]: { $set: action.attr }
          },
          layers: {
            [action.attr.parent]:
            {
              name:
              {
                $apply: (layerName) => {
                  return name ? name : layerName
                }
              }
            }
          }
        }
      });
    case 'GET_SHAPES':
      return layersGetShapes(oldConfig, action.shapes);
    case 'ADD_LAYER':
      return getNewLayers(getLayer(action.layerName), oldConfig);
    case 'IMPORT_MODEL':
      return layersFromModel(action.model,oldConfig)
    case 'DEL_LINK':
      return layersDelLink(oldConfig, action.from, action.to)
    case 'ADD_LINK':
      return layersAddLink(oldConfig, action.from, action.to)
    case 'COPY_LAYERS':
      let newLayers = copyLayers(action.ids, oldLayers, oldConfig)
      return normalize(newLayers, layersSchema)
    case 'DEL_LAYERS':
      return deleLayer(action.ids, oldConfig);
    case 'BUILD_MODULE':
      return buildModule(action.ids, oldConfig)
    // let moduleTempt = {
    //     func: "module",
    //     name: "module",
    //     acts: [{ name: `module`, pars: { name: 'module' }, parsDefault: {} }],
    //     layers: [],
    //     inLayer: {},
    //     outLayer: {},
    //     folded: false
    //   }

    //   return layers2Module(action.ids, oldLayers, [], moduleTempt);
    case 'SHARE_MODULE':
      // let oldLayers = denormalize(oldConfig.result, layersSchema, oldConfig.entities)
      oldLayers = shareModule(action.id, oldLayers)
      return normalize(oldLayers, layersSchema)
    case "FOLDED":
      return foldALayer(oldConfig, action.id)
    default:
      return state
  }
}

const foldALayer = (oldConfig, id) => {
  let newConfig = { ...oldConfig }
  let { layers } = newConfig.entities
  let layer = layers[id]
  layer.folded = !layer.folded
  return newConfig
}

const layersChangePar = (layers, selected, pars, parsDefault) => {

  let selectedLayer = search(layers, selected[0])
  let selectedAct = selectedLayer.acts[selected[1]]
  selectedAct.pars = pars
  selectedAct.parsDefault = parsDefault

  selectedLayer.name = pars.name || selectedLayer.name
  return layers
}

const layersGetShapes = (config, shapes) => {
  // layers.forEach(layer => {
  //   let i = shapes.findIndex(d => d.id == layer.id)
  //   if (i != -1) {
  //     layer.outShape = shapes[i].shape
  //     layer.params = shapes[i].params
  //     shapes.splice(i, 1)
  //   }
  // })
  // return layers
  let newConfig = update(config,{
    entities: {
      layers: {
        $apply: (layers) => {
          shapes.forEach(info => {
            let { shape, params, id } = info
            layers[id].outShape = shape
            layers[id].params = params
          })
          return layers
        }
      }
    }
  })

  return newConfig
}

const getNewLayers = (layer, oldConfig) => {
  let newLayer = { ...layer }
  if (oldConfig.result.length > 0) {
    let befID = oldConfig.result[oldConfig.result.length - 1]
    const checkIn = (id) => {
      let layer = oldConfig.entities.layers[id]
      if (layer.func == 'module') {
        return checkIn(layer.outLayer)
      } else {
        return id
      }
    }

    befID = checkIn(befID)
    newLayer.inputs = [befID]
    // console.info('id',before)
    // console.info('config:', oldConfig)
    // let newConfig = update(oldConfig, {
    //   entities: {
    //     layers: {
    //       [before]: {
    //         outputs: { $push: [layer.id] }
    //       }
    //     }
    //   }
    // })
  }
  return addLayer(newLayer, oldConfig)
}

const layersFromModel = (model, oldConfig) => {
  let layers
  //keras model, created by model.to_json()
  if (model.keras_version) {
    layers = model.config.layers.map(layer => {
      let inputs = layer.inbound_nodes[0] ? layer.inbound_nodes[0].map(d => d[0]) : []
      let pars = {}
      let parsDefault = {}
      //treat input layer seperately
      if (layer.class_name == "InputLayer") {
        layer.class_name = "Input"
        pars.dtype = `"${layer.config.dtype}"`
        let dims = []
        layer.config.batch_input_shape.forEach(dim => {
          if (dim) {
            dims.push(`${dim},`)
          }
        })
        pars.shape = `(${dims.join('')})`
      } else {
        Object.keys(layer.config).forEach(key => {
          if(key =='config'){
            
          }else {
            let value = layer.config[key]
            pars[key]=findString(value)
          }

        })
      }

      return {
        name: layer.name,
        func: layer.class_name,
        id:layer.name,
        layers:[],
        shared:'',
        parent:'',
        // id: `${layer.name}_${layerIndex}`,
        attrs: [{ 
          tag: layer.class_name,
          id:`${layer.name}_${getID()}`,
          parent:layer.name,
          pars, 
          parsDefault 
        }],
        // folded: true,
        inputs,
      }
    })
    layerIndex += 1
  }
  //IR model
  else if (model.node) {
    layers = model.node.map(node => {
      let inputs = node.input || []
      let pars = node.attr ? attr2pars(node.attr) : {}
      return {
        name: node.op,
        func: name2func(node.name),
        id: `${node.name}_${layerIndex}`,
        acts: [{ name: node.op, pars, parsDefault: '' }],
        // folded: true,
        inputs: inputs.map(d => `${d}_${layerIndex}`)
      }
    })
    layerIndex += 1
  }
  //my model
  // layers = layersFromMyModel(model)
  else {
    console.info('my model',model, model.keras_version.keras_version)
    layers=model
  }
  let config = normalize(layers,layersSchema)
  let newConfig;
  if (oldConfig.result.length>0){
    newConfig = update(oldConfig,{
    entities:{
      layers:{$merge: config.entities.layers},
      attrs:{$merge:config.entities.attrs}
    },
    result:{$push:config.result}
  })
  }else{
    newConfig=config
  }
  console.info('new config', newConfig)
  return newConfig
}

// const layersFromMyModel = (model) => {
//   model.forEach(layer => {
//     layer.id += `_${layerIndex}`
//     layer.inputs = layer.inputs.map(input => input += `_${layerIndex}`)
//     if (layer.func == 'module') {
//       layersFromMyModel(layer.layers)
//     }
//   })
// }

//for parsing IR model
const attr2pars = (attr) => {
  let pars = {}
  Object.keys(attr).forEach(k => {
    pars[k] = findString(attr[k])
  })

  return pars
}
const findString = (obj) => {
  if (Array.isArray(obj)) {
    return `(${obj.map(d => findString(d)).join(',')})`
  } else if (typeof (obj) == "boolean") {
    return obj ? "True" : "False"
  } else if (typeof (obj) == "number") {
    return obj.toString()
  } else if (typeof (obj) == "object" && obj) {
    return Object.values(obj).map(d => findString(d)).join(',')
  } else if (obj) {
    return `"${obj.toString()}"`
  } else if (obj == null) {
    return `None`
  }
}
const name2func = (name) => {
  let arr = name.split("_")
  arr.splice(arr.length - 1, 1)
  return arr.map(d => d.charAt(0).toUpperCase() + d.slice(1)).join('')
}

const copyLayers = (ids, oldLayers,config) => {

  oldLayers.forEach(layer => {
    if (layer.func == 'module' && !layer.folded) {
      copyLayers(ids, layer.layers,config)
    } else if (ids.indexOf(layer.id) != -1) {
      let clone = deepCopy(layer, `_copy`)
      // if (oldLayers.indexOf(clone.inputs[0]) == -1) {
      //   clone.inputs = []
      // }
      let newInputs = []
      clone.inputs.forEach((input,i)=>{
        let suf = input.split('_copy')[0]
        if(ids.indexOf(suf)!=-1){
          newInputs.push(input)
        }
      })
      clone.inputs = newInputs
      oldLayers.push(clone)
    }
  })
  return oldLayers
}



// const delLayers = (ids, oldConfig, newLayers) => {
//   oldConfig.forEach(layer => {
//     if (layer.func == 'module' && !layer.folded) {
//       delLayers(ids, layer.layers, newLayers)
//     } else if (ids.indexOf(layer.id) != -1) {
//       // let i = oldConfig.indexOf(layer)
//       // oldConfig.splice(i, 1)
//     } else if (ids.indexOf(layer.id) == -1) {
//       layer.inputs = layer.inputs.filter(d => {
//         return ids.indexOf(d) == -1
//       })
//       newLayers.push(layer)
//     }
//   })
//   return newLayers
// }
const checkFrom = (from, layersObj) => {
  if (layersObj[from].func == 'module') {
    return checkFrom(layersObj[from].outLayer, layersObj)
  }
  return from
}

const checkTo = (to, layersObj) => {
  if (layersObj[to].func == 'module') {
    return checkTo(layersObj[to].inLayer, layersObj)
  }
  return to
}

const layersAddLink = (oldConfig, from, to) => {
  let layersObj = oldConfig.entities.layers
  from = checkFrom(from, layersObj)
  to = checkTo(to, layersObj)

  return update(oldConfig, {
    entities: {
      layers: {
        [to]: {
          inputs: { $push: [from] }
        },
        // [from]: {
        //   outputs: { $push: [to] }
        // }
      }
    }
  })
}

const layersDelLink = (oldConfig, from, to) => {
  let layersObj = oldConfig.entities.layers
  from = checkFrom(from, layersObj)
  to = checkTo(to, layersObj)
  return update(oldConfig,
    {
      entities: {
        layers: {
          [to]: {
            inputs: {
              $apply: (inputs) => {
                let i = inputs.indexOf(from)
                inputs.splice(i, 1)
                return inputs
              }
            }
          },
          // [from]: {
          //   outputs: {
          //     $apply: (outputs) => {
          //       let i = outputs.indexOf(to)
          //       outputs.splice(i, 1)
          //       return outputs
          //     }
          //   }
          // }
        }
      }
    }
  )
}

// const layers2Module = (ids, oldConfig, newLayers, newModule) => {
//   oldConfig.forEach(layer => {
//     if (layer.func == 'module' && !layer.folded) {
//       layer.layers = layers2Module(ids, layer.layers, [], newModule)
//       newLayers.push(layer)
//     } else if (ids.indexOf(layer.id) != -1) {
//       newModule.layers.push(layer)
//     } else {
//       newLayers.push(layer)
//     }
//   })
//   newModule.inLayer = findInLayers(newModule.layers)[0] || {}
//   newModule.outLayer = findOutLayers(newModule.layers)[0] || {}
//   newModule.id = newModule.outLayer.id
//   newModule.outShape = newModule.outLayer.outShape
//   newModule.inputs = newModule.inLayer.inputs || []
//   newLayers.push(newModule)
//   return newLayers
// }


const shareModule = (id, oldConfig) => {
  oldConfig.forEach(layer => {
    if (layer.func == 'module' && !layer.folded) {
      shareModule(id, layer.layers)
    } else if (layer.id == id) {
      layer.shared = id
      let newLayer = deepCopy(layer, '_shared')
      newLayer.inputs.forEach(input => {
        let newInput = getLayer(input.split('_')[0])
        newInput.inputs = []
        newInput.id = `${input}`
        oldConfig.push(newInput)
      })
      newLayer.shared = id
      oldConfig.push(newLayer)
    } else if (layer.inputs.indexOf(id) != -1) {
      layer.inputs.push(`${id}_shared`)
    }
  })
  return oldConfig
}

export const addLayer = (layer, oldConfig) => {
  if (oldConfig.result.length != 0) {
    let denormalizeLayers = denormalize(oldConfig.result, layersSchema, oldConfig.entities)
    denormalizeLayers.push(layer)
    return normalize(denormalizeLayers, layersSchema)
  } else {
    return normalize([layer], layersSchema)
  }

}


export const deleLayer = (ids, oldConfig) => {
  return update(oldConfig, {
    entities: {
      layers: {
        $apply: (layers) => {
          ids.forEach(del => {
            for(let id in layers){
              let layer = layers[id]
              let i = layer.inputs.indexOf(del)
              if (i != -1) {
                layer.inputs.splice(i, 1)
              }
            }
            delete layers[del]
          })
          return layers
        }
      }
    },
    result: {
      $apply: (result) => {
        ids.forEach(id => {
          let i = result.indexOf(id)
          result.splice(i, 1)
        })
        return result
      }
    }
  })
}

export const buildModule = (ids, oldConfig) => {
  let moduleID = `module_${getID()}`
  let layers = ids.map(id => oldConfig.entities.layers[id])
  let inLayer = findInLayers(layers)[0] || {}
  let outLayer = findOutLayers(layers)[0] || {}
  let inputs = inLayer.inputs || []
  // ,outputs=[]
  // inLayers.forEach(inLayer=>{
  //   inputs = inputs.concat(inLayer.inputs)
  // })
  // outLayers.forEach(outLayer=>{
  //   outputs= outputs.concat(outLayer.outputs)
  // })

  const newModule = {
  [moduleID]: {
    id: moduleID,
    func: 'module',
    folded: true,
    name: 'module',
    parent: '',
    layers: ids,
    inLayer: inLayer.id,
    outLayer: outLayer.id,
    inputs,
    // outputs,
    attrs: [`moduleAttr_${getID()}`]
  }
  }
  const moduleAttr = {
    [`moduleAttr_${getID()}`]: {
      id: `moduleAttr_${getID()}`,
      tag: 'module',
      name: 'module',
      parent: moduleID,
      pars: { name: 'module' },
      parsDefault: {}
    }
  }

  ids.forEach((id, index) => {
    let layer = oldConfig.entities.layers[id]
    if (layer.parent != '') {
      let parentID = layer.parent
      let parent = oldConfig.entities.layers[parentID]
      let children = parent.layers
      let i = children.indexOf(id)
      if (i != -1) { children.splice(i, 1) }
      if (children.indexOf(moduleID) == -1) {
        children.push(moduleID)
      }
      layer.parent = moduleID
    } else {
      layer.parent = moduleID
      let { result } = oldConfig
      let i = result.indexOf(id)
      result.splice(i, 1)
      if (index == 0) { result.push(moduleID) }
    }
  })
  // let normalizeModule = normalize([newModule], layersSchema)
  let newConfig = update(oldConfig, {
    entities: {
      layers: {
        $merge: newModule
      },
      attrs: {
        $merge: moduleAttr
      }
    }
  })
  return newConfig

}