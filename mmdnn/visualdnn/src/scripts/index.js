import 'babel-polyfill'

import React from 'react';
import ReactDom from 'react-dom';
import App from './containers/App';
import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import thunkMiddleware from 'redux-thunk'
import { createLogger } from 'redux-logger'
import reducer from './reducers';
import '../style/style';

const loggerMiddleware = createLogger()

function configStore(preloadedState) { 
  return createStore(
    reducer,
    preloadedState,
    applyMiddleware(
      thunkMiddleware,
      loggerMiddleware
    )
  )
}

const store = configStore()

window.onload = function () {
  ReactDom.render(<Provider store={store}>
    <App />
  </Provider>, document.getElementById("app"));
};



