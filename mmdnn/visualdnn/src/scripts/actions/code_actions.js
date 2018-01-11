
import {venv} from '../../assets/const'
export const changeCode = (code) => {
  return {
    type: 'CHANGE_CODE',
    content: code
  }
}

export const runCode = (code) => {
  const fs = require('fs')
  const {exec} = require('child_process')
  // code.replace('callbacks=[None]', 'callbacks=[my_callback]')
  let codeArry = code.split('\n')
  codeArry.unshift(`import requests`)
  let my_callback = `class MyCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        requests.post('http://localhost:3000/on_batch_end',
        data={"batch":batch,"loss":logs["loss"],"acc":logs['acc']})
my_callback = MyCallback()`
  codeArry.splice(-2, 0, my_callback)
  let codeWithCallback = codeArry.join('\n')
  console.info(codeWithCallback)
  return dispatch => {
    //if run on remote server
    // return connection.then(
    //   () => fs.writeFile("code", code, function (err) {
    //     if (err) {
    //       return console.log(err);
    //     }
    //   }
    //   )).then(() => {
    //     return ssh.putFile('code', 'serverForVisualDNN/code')
    //   }).then(() => { console.info('upload code') },
    //     (err) => {
    //       console.info(err)
    //     }).then(() => {
    //       return ssh.execCommand('source qianwen/FCN/venv/bin/activate; python3 serverForVisualDNN/code')
    //     }).then(res => {
    //       console.info(res.stdout)
    //       console.info(res.stderr)
    //       return dispatch(runRes(res.stdout))
    //     }
    //     )

    //if run on local
    return Promise.resolve()
      .then(
      () => fs.writeFile("code.py", codeWithCallback, function (err) {
        if (err) { return console.log(err); }
      })
      ).then(() => {
        // const run = spawn('activate tensorflow && python code')

        // run.stdout.on('data', (data) => {
        //   console.log(`stdout: ${data}`);
        // });

        // run.stderr.on('data', (data) => {
        //   console.log(`stderr: ${data}`);
        // });

        // return run

        return exec(`${venv} code.py`, { env: { 'KERAS_BACKEND': 'tensorflow' }, maxBuffer: 1024 * 1024 },
          function (error, stdout, stderr) {
            if (error) { return console.info(error) }
            console.info(stdout)
            console.info(stderr)
            return dispatch(runRes(stdout))
          })
      })

  }
}



const runRes = (stdOut) => {
  return {
    type: 'RUN_RES',
    res: stdOut,
    isRunning: true
  }
}