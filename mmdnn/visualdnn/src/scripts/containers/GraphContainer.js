import { connect } from "react-redux";
import { normalize, denormalize } from 'normalizr';
import { layersSchema } from '../ulti/schema';
import {
  moveLayers,
  selectLayer,
  changePar,
  delLink, addLink,
  folded,
  delLayers,
  buildModule,
  shareModule,
  copyLayers
} from "../actions/graph_actions";
import GraphView from "../components/GraphView";
import * as dagre from 'dagre';

const mapStateToProps = state => {
  return {
    layers: state.config.result.length>0 ?
      denormalize(state.config.result, layersSchema, state.config.entities) : [],
    config: state.config,
    selectedLayer: state.selectedLayer
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    onMoveLayers: (from, to) => {
      dispatch(moveLayers(from, to))
    },
    // onChangePar: (selected, pars, parsDefault) => {
    //   dispatch(changePar(selected, pars, parsDefault))
    //   dispatch(fetchShapes())
    // },
    onSelectLayer: (id) => {
      dispatch(selectLayer(id))
    },
    onDelLink: (from, to) => {
      dispatch(delLink(from, to))
    },
    onAddLink: (from, to) => {
      dispatch(addLink(from, to))
    },
    onFolded: (id) => {
      dispatch(folded(id))
    },
    onCopyLayers: (ids) => {
      dispatch(copyLayers(ids))
    },
    onDelLayers: (ids) => {
      dispatch(delLayers(ids))
    },
    onBuildModule: (ids) => {
      dispatch(buildModule(ids))
    },
    onShareModule: (ids) => {
      dispatch(shareModule(ids))
    }
  }
}





const GraphContainer = connect(mapStateToProps, mapDispatchToProps)(GraphView);

export default GraphContainer;
