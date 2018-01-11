import { connect } from "react-redux";
import {
  changePar,
  fetchShapes
} from "../actions/par_actions";
import ParBoard from "../components/ParBoard";


const mapStateToProps = state => {
  return {
    attrs: getSelectedAttrs(state.config, state.selectedLayer)
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    onChangePar: (attr) => {
      dispatch(changePar(attr))
    },
    onFetchShapes: () => {
      dispatch(fetchShapes())
    }
  }
}

// const getSelectedAct=(state)=>{
//     let [a,b]= state.selected
//     let {layers} = state
//     let selectedAct = { name: 'None', pars: {}, parsDefault: {} }
//     if (b != -1 && layers != [] ) { selectedAct = layers[a]['acts'][b] }
//     return selectedAct
// }

const ParBoardContainer = connect(mapStateToProps, mapDispatchToProps)(ParBoard);

export default ParBoardContainer;

// const getSelectedAttrs = (layers, id) => {
//   for (let i =0;i<layers.length;i++){
//     let layer = layers[i]
//     if(layer.func=='module'&&!layer.folded){
//       return getSelectedAttrs(layer.layers, id)
//     }else if (layer.id==id){
//       return layer
//     }
//   }
//   return {name:'none', acts:[]}
// }

const getSelectedAttrs = (config, id) => {
  
  if (config.result.length!=0&&id) {
    let attrsID = config.entities.layers[id].attrs
    return attrsID.map((attrID,i) => {
      return config.entities.attrs[attrID]
    })
  }else{
    return [{tag:'None',pars:{},parsDefault:{}}]
  }
}