
import { COMPILE, FIT, TF_CB } from '../../assets/def';
export const dataset = (state = "", action) => {
    switch (action.type) {
        case 'ADD_KERAS_DATASET':
            return action.name;

        default:
            return state;
    }
}

export const setting = (
    setting = {
        compile:{},
        run: {}
    },
    action) => {
        switch (action.type) {
            case 'ADD_COMPILE':
                return {
                    ...setting,
                    compile: action.compile
                }
            case 'ADD_RUN':
                return {
                    ...setting,
                    run: action.run
                }
            default:
                return setting;
        }
}

export const selectedLayer = (selectedLayer = null, action) => {
    switch (action.type) {
        case 'SELECT_LAYER':
            return action.selected;
        default:
            return selectedLayer
    }
}

