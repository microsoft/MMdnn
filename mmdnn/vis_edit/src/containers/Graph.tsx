import Graph from '../components/Graph';
import * as actions from '../actions/';
import { StoreState } from '../types/index';
import { connect, Dispatch } from 'react-redux';

export function mapStateToProps(state:StoreState, ownProps:{width:number}) {
    return {
        nodes: state.model.node, 
        selectedLayer: state.selectedLayer,
        width:ownProps.width
    };
}

export function mapDispatchToProps(dispatch: Dispatch<actions.ImportModelAction>) {
    return {
        
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Graph);