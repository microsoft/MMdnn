const express = require('express')
const app = express()
app.use(express.static('public'))
app.get('/', (req, res) => res.sendFile('index.html', {"root": __dirname}))

app.listen(8080, () => console.log('vis is listening on port 8000!'))