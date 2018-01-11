import { deepCopy } from './ulti_layers'
import { normalize, denormalize } from 'normalizr'
import { layersSchema } from '../ulti/schema'
export function generateCode(state) {
    //code start
    let code = ['import keras',
        'from keras import layers',
        'from keras.models import Model',
        'from keras import backend as K']
    //import dataset

    if (state.dataset != '') {
        let inputLayer = state.layers.filter(d => d.func == 'Input')[0]
        let inputShape = inputLayer ? inputLayer.acts[0].pars.shape : ''
        inputShape = "," + inputShape.replace(/\(|\,\)/g, "")
        code.push(`from keras.datasets import ${state.dataset}`)
        code.push(`(x_train, y_train), (x_test, y_test) = ${state.dataset}.load_data()`)
        if (state.dataset == 'mnist') {
            code.push(`x_train = x_train.reshape(x_train.shape[0] ${inputShape})`)
            code.push(`x_test = x_test.reshape(x_test.shape[0] ${inputShape})`)
            code.push(`x_train = x_train.astype('float32')/255`)
            code.push(`x_test = x_test.astype('float32')/255`)
            code.push(`y_train = keras.utils.to_categorical(y_train, 10)`)
            code.push(`y_test = keras.utils.to_categorical(y_test, 10)`)
        } else if (state.dataset == "imdb") {
            let features = state.layers.filter(d => d.func == "Embedding")[0].acts[0].pars.input_dim
            let steps = state.layers.filter(d => d.func == "Input")[0].acts[0].pars.shape.replace(/\(|\,\)/g, "")
            code.splice(1, 0, `from keras.preprocessing import sequence`)
            code.push(`(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=${features})`)
            code.push(`x_train = sequence.pad_sequences(x_train, maxlen=${steps})`)
            code.push(`x_test = sequence.pad_sequences(x_test, maxlen=${steps})`)
        }

    }

    //model
    code = code.concat(generateModel(state.config))

    let { compile, run } = state.setting
    let { fit, callbacks } = run
    if (!_.isEmpty(compile)) {
        let compilePar = ''
        Object.keys(compile).map(key => {
            compilePar += `${key}=${compile[key]}, `
        })
        code.push(`model.compile(${compilePar})`)
    }
    saveCode(code)
    if (!_.isEmpty(run)) {
        let fitPar = Object.keys(fit).map(key => `${key}=${fit[key]}`).join(',')
        let tf_cb = Object.keys(callbacks).map(key => `${key}=${callbacks[key]}`).join(',')
        code.push(`tf_cb=keras.callbacks.TensorBoard(${tf_cb})`)
        code.push(`model.fit(${fitPar},  callbacks=[my_callback,tf_cb])`)
    }
    return code.join('\n')
}

export function findInLayers(layers) {
    let inLayers = []
    let ids = layers.map(d => d.id)
    layers.forEach(layer => {
        let inFlag = false
        layer.inputs.forEach(input => {
            if (ids.indexOf(input) == -1) {
                inFlag = true
            }
            if (inFlag) { inLayers.push(layer) }
        })
    })
    return inLayers
}

export function findOutLayers(layers) {
    let outLayers = []
    let ids = []
    layers.forEach(d => {
        d.inputs.forEach(input => {
            ids.push(input)
        })
    })
    layers.forEach(layer => {
        if (ids.indexOf(layer.id) == -1) {
            outLayers.push(layer)
        }
    })
    return outLayers
}

function generateModel(config) {
    let code = [],
        modelInputs = [],
        { newOrder, refineOrder } = resortLayers(config),
        modelOutputs = findOutLayers(denormalize(newOrder, layersSchema, config.entities)).map(d => d.id),
        layers = denormalize(refineOrder, layersSchema, config.entities)

    // modules = layers.filter(layer => layer.func == "module")
    layers.forEach(layer => {
        if (layer.inputs.length == 0) {
            modelInputs.push(layer.id)
        }
        if (layer.func == "module" && layer.shared) {
            //create the shared module
            if (layer.shared == layer.id) {
                let submodel = [
                    `def create_${layer.id}_module(input_dim):`,
                    `    in_layer=layers.Input(shape=input_dim)`
                ]
                let inLayer = layer.inLayer
                let outLayer = layer.outLayer
                let subDict = normalize(layer.layers, layersSchema)
                let orders = sortDag(deepCopy(subDict.entities.layers,''),[inLayer])
                let layersInModule = denormalize(orders, layersSchema,config.entities)
                layersInModule.forEach(layerSub => {
                    let line = generateLine(layerSub)
                    if (layerSub.id == inLayer) {
                        line = line.replace(/\([^\)]+\)$/, "(in_layer)")
                    } else if (layerSub.id == outLayer) {
                        line = line.replace(/[^=]+\=/, "out_layer=")
                    }
                    submodel.push(`    ${line}`)
                })
                submodel.push(`    return Model(in_layer, out_layer)`)
                code = code.concat(submodel)

                code.push(`${layer.id}=create_${layer.id}_module(${inLayer}.input_shape)`)
                //use the shared module
                code.push(`proc_${layer.id}=${layer.id}(${layer.inputs})`)
            } else {
                //use the shared module
                code.push(`proc_${layer.id}=${layer.shared}(${layer.inputs})`)
            }
        } else if (layer.func == "module" && !layer.shared) {
            layer.layers.forEach(layer => {
                code.push(generateLine(layer))
            })
        } else {
            let clone = deepCopy(layer, '')
            for (let i = 0; i < clone.inputs.length; i++) {
                let input = clone.inputs[i]
                if (input.includes('_shared')) {
                    let parent = input.split('_shared')[0]
                    let iParent = clone.inputs.indexOf(parent)
                    clone.inputs[iParent] = `proc_${parent}`
                    clone.inputs[i] = `proc_${input}`
                }
            }
            let line = generateLine(clone)
            // if (modules[0] && layer.inputs[0] == modules[0].outLayer.id) {
            //     line = line.replace(/\([^\)]+\)$/, `(proc_${modules[0].outLayer.id})`)
            // }
            code.push(line)
        }
    })
    code.push(`model = Model(inputs=[${modelInputs.join(',')}], outputs=[${modelOutputs.join(',')}])`)
    return code
}

function generateLine(layer) {
    let allPars = ''
    layer.attrs.forEach(attr => {
        // let pars = attr.pars
        // Object.keys(pars).forEach(key => {
        //     if (key != 'name') {
        //         allPars += `${key}=${pars[key]}, `
        //     }
        // })
        for (let k in attr.pars) {
            if (k != 'name') {
                allPars += `${k}=${attr.pars[k]}, `
            }
        }
        // for (let k in attr.parsDefault){
        //     allPars += `${k}=${attr.parsDefault[k]}, `
        // }
    })

    let line;
    //input layer
    if (layer.inputs.length == 0) {
        line = `${layer.id}=layers.${layer.func}(${allPars} name='${layer.id}')`
    }
    //merge layer
    else if (layer.inputs.length > 1) {
        let axis = (layer.name == "concat") ? `axis=${layer.acts[0].pars.axis},` : ''
        if (layer.func == "Concatenate") { layer.func = "concatenate" }//seems a bug in keras
        line = `${layer.id}=layers.${layer.func}(inputs=[${layer.inputs.join(',')}],${axis} name='${layer.id}')`
    }
    //ordinary layer
    else {
        line = `${layer.id}=layers.${layer.func}(${allPars} name='${layer.id}')(${layer.inputs[0]})`
    }

    return line
}

function resortLayers(config) {
    let layerDict = config.entities.layers
    let queue = []
    let layersObj = {}
    for (let id in config.entities.layers) {
        let layer = layerDict[id]
        if (layer.layers.length == 0) {
            layersObj[id] = deepCopy(layerDict[id], '')
        }
    }
    for (let id in layersObj) {
        if (layersObj[id].inputs.length == 0) {
            queue.push(id)
        }
    }
    let newOrder=sortDag(layersObj,queue)
    let refineOrder = [...newOrder]
    newOrder.forEach(id => {
        let parentID = layerDict[id].parent
        if (parentID != '' && layerDict[parentID].shared) {
            let i = refineOrder.indexOf(id)
            // let parent = layerDict[parentID]
            // if(parent.shared==parentID&&refineOrder.indexOf(parentID)==-1){
            //     refineOrder.splice(i, 1, parentID)
            // }else{
            //     refineOrder.splice(i,1)
            // }
            if (refineOrder.indexOf(parentID) == -1) {
                refineOrder.splice(i, 1, parentID)
            } else {
                refineOrder.splice(i, 1)
            }
        }
    })
    // console.info('new order', refineOrder)
    return { newOrder, refineOrder }
}

function sortDag(layersObj,queue) {
    let newOrder = []
    // let cloneObj = {...layersObj}
    while (queue.length != 0) {
        // console.info('queue', queue)
        let v = queue.shift()
        newOrder.push(v)
        delete layersObj[v]
        // console.info('delete:', v)
        for (let id in layersObj) {
            let inputs = layersObj[id].inputs
            let i = inputs.indexOf(v)
            if (i != -1) {
                inputs.splice(i, 1)
                if (inputs.length == 0) {
                    queue.push(id)
                    // console.info('add to queue:', id)
                }
            }

        }
    }
    return newOrder
}
function saveCode(code) {
    const fs = require('fs')
    let lib = `import json`
    let printCommand = `shapes = []
for layer in model.layers:
    shape=list()
    for dim in layer.output_shape:
        if(dim==None):
            shape.append("none")
        else:
            shape.append(int(dim))
    shapes.append(json.dumps({"id":layer.name, "shape":shape, "params": int(layer.count_params())}))
print(shapes)`
    let loggerCode = [lib, ...code, printCommand].join('\n')
    fs.writeFile("logger.py", loggerCode, function (err) {
        if (err) {
            return console.log(err);
        }
    });

}