export const COLORS = [
  '#87f7cf',
  '#f7f494',
  '#72ccff',
  '#f7c5a0',
  '#fc97af',
  '#d4a4eb',
  '#d2f5a6',
  '#76f2f2',
];

let uid = 0
export const getID = () => {
  return Date.now().toString().slice(8,13)
}

export const COMPILE = {
  optimizer: [
    'keras.optimizers.Adadelta()',
    'keras.optimizers.RMSprop()',
    'keras.optimizers.SGD()',
    'keras.optimizers.Adam()'
  ],
  loss: ['keras.losses.categorical_crossentropy',
    'keras.losses.mean_squared_error',
    'keras.losses.mean_absolute_error',
    'keras.losses.binary_crossentropy'
  ],
  metrics: `['accuracy']`,
  loss_weights: 'None',
  sample_weight_mode: 'None'
}
export const FIT = {
  x: 'x_train',
  y: 'y_train',
  batch_size: 32,
  epochs: 1,
  verbose: 1,
  // callbacks: 'None',
  validation_split: 0.0,
  validation_data: '(x_test, y_test)',
  shuffle: 'True',
  class_weight: 'None',
  sample_weight: 'None',
  initial_epoch: 0
}

export const TF_CB = {
  log_dir: `"./logs"`,
  histogram_freq: 0,
  batch_size: 32,
  write_graph: `True`,
  write_grads: `False`,
  write_images: `False`,
  embeddings_freq: 0,
  embeddings_layer_names: `None`,
  embeddings_metadata: `None`
}

export const dictCate = {
  Input: ['Input'],
  Core: [
    'Activation',
    "Dense",
    "Dropout",
    "Flatten",
    "Reshape",
    'RepeatVector',
    "Lambda"
  ],
  Conv: ["Conv1D", "Conv2D", "Conv3D"],
  Pooling: [
    "MaxPooling1D",
    "MaxPooling2D",
    "MaxPooling3D",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
  ],
  Recurrent: [
    "SimpleRNN",
    "GRU",
    "LSTM"
  ],
  Merge: [
    "Add",
    "Multiply",
    "Average",
    "Dot",
    "Maximum",
    "concatenate"
  ],
  Embedding: ['Embedding'],

  Wrapper: ['TimeDistributed', 'Bidirectional']
}

export const dictLayer = {
  Input: ["Input"],

  Conv1D: ['Filters', 'Bias', 'Acti'],
  Conv2D: ['Filters', 'Bias', 'Acti'],
  Conv3D: ['Filters', 'Bias', 'Acti'],

  MaxPooling1D: ['Pooling'],
  MaxPooling2D: ['Pooling'],
  MaxPooling3D: ['Pooling'],

  AveragePooling1D: ['Pooling'],
  AveragePooling2D: ['Pooling'],
  AveragePooling3D: ['Pooling'],

  Dense: ['Kernel', 'Bias', 'Acti'],
  Flatten: ['Flatten'],
  Dropout: ['Dropout'],
  Activation: ['Acti'],
  Reshape: ["Reshape"],
  RepeatVector: ['RepeatVector'],
  Lambda: ['Lambda'],

  Add: ['Merge'],
  Multiply: ['Merge'],
  Average: ['Merge'],
  Dot: ['Merge'],
  Maximum: ['Merge'],
  concatenate: ['Concat'],

  SimpleRNN: ['Recurrent', 'Initial_R', "Regularize_R", "Constraint_R"],
  GRU: ['Recurrent', 'Initial_R', "Regularize_R", "Constraint_R"],
  LSTM: ['Recurrent', 'Initial_R', "Regularize_R", "Constraint_R"],

  Embedding: ['Embedding'],

  TimeDistributed: ['TimeDistributed'],
  Bidirectional: ['Bidirectional']
}
export const dictAttr = {
  Filters: {
    need: {
      filters: '',
      kernel_size: '',
    },
    default: {
      dilation_rate: '(1,1)',
      strides: '(1,1)',
      padding: 'valid'
    }
  },
  Bias: {
    need: {

    },
    default: {
      use_bias: 'True', bias_initializer: 'zeros', bias_regularizer: 'None', bias_constraint: 'None'
    }
  },
  Kernel: {
    need: { units: '' },
    default: { kernel_initializer: 'glorot_uniform', kernel_constraint: 'None' }
  },
  Acti: {
    need: {},
    default: {
      activation: [`None`, `"sigmoid"`, `"relu"`, `"tanh"`, `"softmax"`],
      activity_regularizer: 'None'
    }
  },
  Pooling: {
    need: { pool_size: '' },
    default: { strides: 'None', padding: 'valid' }
  },
  Dropout: {
    need: { rate: '' },
    default: { noise_shape: 'None', seed: 'None' }
  },
  RepeatVector: {
    need: { n: '' }
  },


  Input: {
    need: { shape: '' },
    default: { dtype: 'None' }
  },
  Merge: {},
  Concat: { need: { inputs: '' }, default: { axis: '-1' } },
  Reshape: { need: { target_shape: '' } },
  Flatten: {},
  Lambda: {
    need: { function: '' },
    default: {
      output_shape: 'None', mask: 'None', arguments: 'None'
    }
  },
  Recurrent: {
    need: { units: '' },
    default: {
      activation: 'tanh',
      use_bias: "True",
    }

  },
  Initial_R: {
    default: {
      kernel_initializer: 'glorot_uniform',
      recurrent_initializer: 'orthogonal',
      bias_initializer: 'zeros',
    }
  },
  Regularize_R: {
    default: {
      dropout: "0.0",
      recurrent_dropout: "0.0",
      kernel_regularizer: "None",
      recurrent_regularizer: "None",
      bias_regularizer: "None",
      activity_regularizer: "None"
    }
  },
  Constraint_R: {
    default: {
      kernel_constraint: "None",
      recurrent_constraint: "None",
      bias_constraint: 'None',
    }
  },

  Embedding: {
    need: {
      input_dim: '',
      output_dim: '',
    },
    default: {
      input_length: 'None',
      embeddings_initializer: 'uniform',
      embeddings_regularizer: 'None',
      activity_regularizer: 'None',
      embeddings_constraint: 'None',
      mask_zero: 'False',
    }
  },

  TimeDistributed: { need:{layer: ''} },
  Bidirectional: { need:{ layer: ''},default:{ merge_mode: 'concat', weights: 'None'} }
}

Object.freeze(dictAttr)
Object.freeze(dictLayer)

export const getLayer = (name) => {
  let layer = {
  name,
  func : name,
  id : `${name}_${getID()}`,
  parent:'',
  layers : [],
  inputs:[],
  // outputs:[]
}
  layer.attrs = dictLayer[name].map((d, i) => {
    // let pars = i==0?{name:name}:{}, parsDefault = {}
    // let allPars = { ...dictAct[d] }
    // Object.keys(allPars).forEach(k => {
    //   if (allPars[k] == '') { pars[k] = allPars[k] }
    //   else { parsDefault[k] = allPars[k] }
    // })
    // return {
    //   name: d,
    //   pars,
    //   parsDefault
    // }
    // let pars = {}, parsDefault={}, allPars=dictAttr[d]
    // for (k in allPars){
    //   if (allPars[k]===''){
    //     pars[k]=allPars[k]
    //   }else{
    //     parsDefault[k]=allPars[k]
    //   }
    // }
    let attr = {
      id: `${d}_${getID()}`,
      tag: d,
      parent:layer.id,
      pars:dictAttr[d].need||{},
      parsDefault:dictAttr[d].default||{}
    }
    if (i == 0) { attr.pars = Object.assign({ name: name }, attr.pars) }
    return attr
  })
  return layer
}

export const KERAS_DATASETS = ['cifar10', 'cifar100', 'imdb', 'reuters', 'mnist', 'boston_housing']
export const KERAS_MODELS = ['VGG16', 'VGG19', 'InceptionV3']



export const getLayerColor = layer => {
  // let a = ALL_LAYERS.indexOf(layer)%COLORS.length;
  // let b = LAYERS.indexOf(layer)%COLORS.length;
  // let c = CNN_LAYERS.indexOf(layer)%COLORS.length;
  // return COLORS[c]||COLORS[b]||COLORS[a];
  let index = Object.keys(dictLayer).indexOf(layer) % COLORS.length;
  return COLORS[index] || '#87f7cf'
};

export const getLayerCode = layer => `model.add(${layer}())\n`
