// src/reducers/index.tsx

import { AllActions } from '../actions';
import { StoreState } from '../types';
import { IMPORT_MODEL, SELECT_LAYER } from '../constants';

export function reducer(state: StoreState, action: AllActions): StoreState {
  switch (action.type) {
    // case INCREMENT_ENTHUSIASM:
    
    //   return { ...state, enthusiasmLevel:state.enthusiasmLevel+1 };
    // case DECREMENT_ENTHUSIASM:
    //   return { ...state, enthusiasmLevel:state.enthusiasmLevel-1 };
    case IMPORT_MODEL:  
      return { ...state, model:action.model}
    case SELECT_LAYER:
      return { ...state, selectedLayer: action.name}
    default:
      return state;
  }
}