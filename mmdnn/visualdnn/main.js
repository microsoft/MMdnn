// const electron = require('electron')
// const app = electron.app
// const BrowserWindow = electron.BrowserWindow
// const Menu = electron.Menu
const { app, BrowserWindow, Menu,dialog } = require('electron')
const path = require('path')
const url = require('url')

// require('electron-reload')(__dirname, {
//   electron: path.join(__dirname, 'node_modules', '.bin', 'electron')
// });

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 900,
    backgroundColor: 'black'
  })

  // and load the index.html of the app.
  mainWindow.loadURL(url.format({
    pathname: path.join(__dirname, 'index.html'),
    protocol: 'file:',
    slashes: true
  }))

  // Open the DevTools.
  // mainWindow.webContents.openDevTools()

  // Emitted when the window is closed.
  mainWindow.on('closed', function () {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })

  var menuTemplate = [{
    label: app.getName(),
    submenu: [
      { role: 'about' },
      { type: 'separator' },
      { role: 'quit' }
    ]
  }, {
    label: "Sketch",
    submenu: [
      { label: "New", accelerator: "CmdOrCtrl+N" },
      { type: "separator" },
      {
        label: "Open",
        accelerator: "CmdOrCtrl+O"
      },
      { type: "separator" },
      {
        label: "Save",
        accelerator: "CmdOrCtrl+S",
        // click: function () {
        //   const fs = require("fs")
        //   const timestamp = Date.now()
        //   const folderPath = `./sketches/${timestamp}`
        //   fs.mkdir(folderPath, err => { if (err) throw err })

        //   const printOptions = {
        //     printBackground: true,
        //     landscape: true,
        //     marginsType: 1
        //   }
        //   mainWindow.webContents.printToPDF(printOptions, (error, data) => {
        //     if (error) throw error
        //     fs.writeFile(folderPath + '/screen.pdf', data, (error) => { if (error) throw error })
        //   })

        //   mainWindow.webContents.send('save', {
        //     folder: folderPath
        //   });
        // }
      },
      {
        label: "Save As",
        accelerator: "Shift+CmdOrCtrl+S",
        click:()=>{
          const fs = require("fs")
          console.log(dialog.showSaveDialog(
            {properties: ['openFile', 'openDirectory']},
            {defaultPath:'./sketches'}
          ))
        }
      },
      { label: "Save All", accelerator: "Option+CmdOrCtrl+S" },
      { type: "separator" },
      { label: "Close", accelerator: "CmdOrCtrl+W" },
      { label: "Close All", accelerator: "Option+CmdOrCtrl+W" },
    ]
  }, {
    label: "Edit",
    submenu: [
      { label: "Undo", accelerator: "CmdOrCtrl+Z", selector: "undo:" },
      { label: "Redo", accelerator: "Shift+CmdOrCtrl+Z", selector: "redo:" },
      { type: "separator" },
      { label: "Cut", accelerator: "CmdOrCtrl+X", selector: "cut:" },
      { label: "Copy", accelerator: "CmdOrCtrl+C", selector: "copy:" },
      { label: "Paste", accelerator: "CmdOrCtrl+V", selector: "paste:" },
      { label: "Select All", accelerator: "CmdOrCtrl+A", selector: "selectAll:" }
    ]
  }, {
    label: 'Window',
    role: 'window'
  }, {
    label: 'View',
    submenu: [
      {
        label: "Toggle Layout",
        click: function () {
          mainWindow.webContents.send('toggleLayout', null);
        }
      },
      {
        type: 'separator'
      },
      {
        role: 'reload'
      },
      {
        role: 'toggledevtools'
      },
      {
        type: 'separator'
      },
      {
        role: 'resetzoom'
      },
      {
        role: 'zoomin'
      },
      {
        role: 'zoomout'
      },
      {
        type: 'separator'
      },
      {
        role: 'togglefullscreen'
      }
    ]
  },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(menuTemplate));
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) {
    createWindow()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.

// ipcMain.on('async', (event, arg)=>{
//   console.info(arg);
//   event.sender.send('async-reply', 2)
// })

// exports.pong = arg => {
//   console.log(arg)
// }