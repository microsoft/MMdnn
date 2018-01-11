import {venv} from '../../assets/const'
export const changePar = (attr) => {
  return {
    type: 'CHANGE_PAR',
    attr
  }
}

//async action
export const fetchShapes = () => {
  const {exec} = require('child_process')
  return dispatch => {
    // //fetch on remote server

    // return connection.then(() => {
    //   return ssh.putFile('logger', 'serverForVisualDNN/logger')
    // }).then(() => { console.info('upload logger') },
    //   (err) => {
    //     console.info(err)
    //   }).then(() => {
    //     return ssh.execCommand('source qianwen/FCN/venv/bin/activate; python3 serverForVisualDNN/logger')
    //   }).then(res => {
    //     console.info(res.stdout)
    //     console.info(res.stderr)
    //     return dispatch(getShapes(res.stdout))
    //   }
    //   )

    // run on local to fetch shapes and debug
    return exec(`${venv} logger.py`, { env: { 'KERAS_BACKEND': 'tensorflow' } },
      function (error, stdout, stderr) {
        console.info(stdout)
        console.info(stderr)
        return dispatch(getShapes(stdout))
      })

  }
}

const getShapes = (stdout) => {
  let reg = /\{([^}]+)\}/g;
  let arr;
  let shapes = []
  while ((arr = reg.exec(stdout)) !== null) {
    // result.push(content[arr.index]);
    let json = JSON.parse(arr[0])
    // console.info(json)
    shapes.push(json)
    // index+=1
  }
  return {
    type: 'GET_SHAPES',
    shapes
  }
}