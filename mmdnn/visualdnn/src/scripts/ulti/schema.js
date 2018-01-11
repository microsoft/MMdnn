const { schema, normalize, denormalize } = require('normalizr')
const update = require('react-addons-update')


export const attrSchema = new schema.Entity('attrs')
export const layerSchema = new schema.Entity('layers')
export const layersSchema = new schema.Array(layerSchema);
layerSchema.define({ layers: layersSchema, attrs: [attrSchema] });
// const config = new schema.Entity('config', { layersSchema });


export const addLayer = (layer, oldConfig) => {
    if (oldConfig.result) {
        let denormalizeLayers = denormalize(oldConfig.result, layersSchema, oldConfig.entities)
        denormalizeLayers.push(layer)
        return normalize(denormalizeLayers, layersSchema)
    }else{
        return normalize([layer], layersSchema)
    }

}


export const deleLayer = (ids, oldConfig) => {
    return update(oldConfig,{
        entities:{
            layers:{$apply:(layers)=>{
                ids.forEach(id=>{
                    //since delete is not that often, ignore cache assumption
                    //leave the layer.layers and layer.attrs in the cache
                    delete layers[id]
                })
                for (let id in layers){
                    let layer = layers[id]
                    layer.inputs = layer.inputs.filter(input=>ids.indexOf(input)==-1)
                }
                return layers
            }
        }},
        result:{
            $apply:(result)=>{
                ids.forEach(id=>{
                    let i = result.indexOf(id)
                    result.splice(i,1)
                })
                return result
            }
        }
    })
}

export const compressLayer = (newID, ids, oldLayers) => {
    let newLayers = update(oldLayers, {
        entities: {
            layers: {
                $apply: (layers) => {
                    // console.info('layres:',layers)
                    layers[newID] = {
                        id: newID,
                        layers: ids.map(id => layers[id])
                    }
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
                result.push(newID)
                return result
            }
        }
    })
    return newLayers

}

// const input = [
//         {
//             id:'a1',
//             name:'first',
//             attrs:[{id:'attr1'}, {id:'attr2'}],
//             layers: [{ id:'a4', name:'fourth', layers: [{ id:'a5', name:'fifth',layers: [] }] }]
//         }, {
//             id:'a2',
//             name:'second',
//             layers: []
//         }, {
//             id:'a3',
//             name:'third',
//             layers: []
//         }
//     ]
// let normalizeLayers = normalize(input, layersSchema)
// console.log('origi:',JSON.stringify(normalizeLayers, null, 2))
// let insertLayer = {
//     id:'1a0',
//     name:'tenth',
//     layers:[{id:'9a', name:'ninth',layers:[]}]
// }
// let insertLayers = addLayer(insertLayer, normalizeLayers)
// console.log('after inserting:',JSON.stringify(insertLayers, null, 2))

// let deleLayers = deleLayer('a3', insertLayers)
// console.info('after delete:', JSON.stringify(deleLayers, null,2))

// let compressLayers = compressLayer('11a', ['a1','a2'], deleLayers)
// console.info('after compress:', JSON.stringify(compressLayers, null,2))



