import SideBar from '../components/SideBar';
import * as actions from '../actions/';
import { StoreState } from '../types/index';
import { connect, Dispatch } from 'react-redux';

export function mapStateToProps(state:StoreState, ownProps:{width:number}) {
    return {
        nodes:state.model.node,
        width:ownProps.width
    };
}

export function mapDispatchToProps(dispatch: Dispatch<actions.AllActions>) {
    return {
        onImportModel: (json:any) => {dispatch(actions.importModel(json))},
        onSelectLayer: (name:string) => {dispatch(actions.selectLayer(name))}
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(SideBar);