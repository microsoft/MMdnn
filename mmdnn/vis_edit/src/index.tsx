import * as React from 'react';
import * as ReactDOM from 'react-dom';
import App from './components/App';
import { createStore } from 'redux';
import { reducer } from './reducers/index';
import { StoreState } from './types/index';
import { Provider } from 'react-redux';
import registerServiceWorker from './registerServiceWorker';
import './index.css';

import 'antd/dist/antd.css';


let initState:StoreState = {model:{node:[]}, selectedLayer:undefined}
const store = createStore<StoreState>(reducer, initState );

ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById('root') as HTMLElement
);
registerServiceWorker();
