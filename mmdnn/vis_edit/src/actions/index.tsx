import * as constants from '../constants'
import {Model} from "../types"

// export interface IncrementEnthusiasmAction {
//     type: constants.INCREMENT_ENTHUSIASM;
// }

// export interface DecrementEnthusiasmAction {
//     type: constants.DECREMENT_ENTHUSIASM;
// }


// export function incrementEnthusiasm(): IncrementEnthusiasmAction {
//     return {
//         type: constants.INCREMENT_ENTHUSIASM
//     }
// }

// export function decrementEnthusiasm(): DecrementEnthusiasmAction {
//     return {
//         type: constants.DECREMENT_ENTHUSIASM
//     }
// }


// 
export interface ImportModelAction {
    type: constants.IMPORT_MODEL,
    model:Model

}
export function importModel(json:Model):ImportModelAction{
    return {
        type:constants.IMPORT_MODEL,
        model:json
    }
}
export interface SelectLayerAction {
    type: constants.SELECT_LAYER,
    name:string
}
export function selectLayer(name: string):SelectLayerAction{
    return {
        type:constants.SELECT_LAYER,
        name
    }
}

// export type EnthusiasmAction = IncrementEnthusiasmAction | DecrementEnthusiasmAction
export type AllActions = ImportModelAction|SelectLayerAction
