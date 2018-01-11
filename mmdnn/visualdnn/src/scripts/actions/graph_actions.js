export const delLink = (from, to) => {
  return {
    type: 'DEL_LINK',
    from,
    to
  }
}

export const addLink = (from, to) => {
  return {
    type: 'ADD_LINK',
    from,
    to
  }
}

export const folded = (id) => {
  return {
    type: `FOLDED`,
    id
  }
}

export const selectLayer = (id) => {
  return {
    type: 'SELECT_LAYER',
    selected: id
  }
}

export const copyLayers = (ids) => {
  return {
    type: "COPY_LAYERS",
    ids
  }
}

export const delLayers = (ids) => {
  return {
    type: 'DEL_LAYERS',
    ids
  }
}

export const buildModule =(ids)=>{
  return{
    type:'BUILD_MODULE',
    ids
  }
}

export const shareModule =(ids)=>{
  return{
    type:'SHARE_MODULE',
    id:ids[0]
  }
}