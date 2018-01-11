export const deepCopy=(layer,suffix)=>{
  if(typeof(layer)=='string'||typeof(layer)=='number'){
    return layer
  }
  let clone={};
  Object.keys(layer).map(key=>{
    switch(typeof(layer[key])){
      case 'object':
        if(Array.isArray(layer[key])){
          if(key=='inputs'){
            clone[key] = layer[key].map(d=>d+suffix)
          }else{
            clone[key]=layer[key].map(d=>deepCopy(d,suffix ))
          }
        }
        else{
          clone[key] = deepCopy(layer[key],suffix)
        }
        break
      case 'string':
        if(key=='id'||key=='inLayer'||key=='outLayer'){
          clone[key] = layer[key]+suffix
        }else if(key=='parent'&&layer[key]!=''){
          clone[key] = layer[key]+suffix
        }else{
          clone[key] = layer[key]
        }
        break
      default:
        clone[key] = layer[key]
        break      
    }
  })
  return clone
}

export const search=(layers, id)=>{
    for(let i=0;i<layers.length;i++){
      let layer = layers[i]
      if(layer.func=='module'&!layer.folded){
        return search(layer.layers, id)
      }else if(layer.id==id){
        return layer
      }
    }
  }