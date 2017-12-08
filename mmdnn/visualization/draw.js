let json;
const nodeW = 150, nodeH = 30;
const filePath = "model.json";
const miniW = window.innerWidth * 0.12
const miniH = (window.innerHeight-85) * 0.98
let rankBT = true


/*hanle import model*/
const handleFileSelect = (evt) => {
    let file = evt.target.files[0]
    let reader = new FileReader()
    reader.onload = e => {
        // console.info(JSON.parse(e.target.result))
        draw(JSON.parse(e.target.result))
        json = JSON.parse(e.target.result)
    }
    reader.readAsText(file)
}
document.getElementById('importModel').addEventListener('change', handleFileSelect, false);
//default, draw the model.json file
// window.onload = () => {
//     d3.json(filePath, (error, json) => {
//         if (error) throw error;
//         // console.info(json)
//         draw(json)
//     })
// }
function reverse(){
    rankBT = !rankBT;
    draw(json)
}

//generate dag
getDag = (layers, mode, margin) => {
    let g = new dagre.graphlib.Graph();
    g.setGraph({ranksep:20, marginx:margin, marginy:margin, rankdir:rankBT?'BT':'TB'});
    g.setDefaultEdgeLabel(() => { return {}; });
    layers.forEach(layer => {
        label = mode == "IR" ? `${layer.name}|${layer.op}` : `${layer.name}:${layer.class_name}`
        g.setNode(layer.name, { label: label, width: label.split('/').pop().length * 8, height: nodeH })
        //IR model or keras model
        if (mode == "IR" && layer.input) {
            layer.input.forEach(input => {
                g.setEdge(input, layer.name)
            })
        } else if (mode == "keras" && layer.inbound_nodes.length > 0) {
            inputs = layer.inbound_nodes[0]
            inputs.forEach(input => {
                g.setEdge(input[0], layer.name)
            })
        }
    })
    dagre.layout(g)
    let nodes = [], edges = []
    g.nodes().forEach((v) => {
        if (g.node(v)) {
            nodes.push(g.node(v))
        }
    })
    g.edges().forEach((e) => {
        edges.push(g.edge(e).points)
    });
    let height = g.graph().height,
        width = g.graph().width
    return { nodes, edges, height, width }
}
//path generator
const arrows = (points) => {
    let len = points.length
    if (len == 0) { return }
    let start = `M ${points[0].x} ${points[0].y}`
    let vias = [];
    for (let i = 0; i < len - 2; i += 2) {
        let cPath;
        cPath = [...Array(3).keys()].map(k => `${points[i + k].x} ${points[i + k].y}`)
        cPath = 'C'.concat(cPath)
        vias.push(`M ${points[i].x} ${points[i].y} ${cPath}`)
    }
    vias = vias.join(' ')
    // let arrowL = `M ${points[len - 1].x} ${points[len - 1].y} l -7 -7`
    // let arrowR = `M ${points[len - 1].x} ${points[len - 1].y} l  7 -7`
    let pathData = `${start}  ${vias}`
    return pathData
}

const OPs = []
const COLORs = d3.schemeCategory20

// select a layer
const selectLayer = (layer, info, mode) => {
    name = layer.label.split("|")[0]
    layerInfo = info.filter(i => i.name == name)[0]
    if (mode == "IR") {
        config = layerInfo.attr
        name = layerInfo.name
    } else if (mode == "keras") {
        config = layerInfo.config
        name = layerInfo.label
    }
    config = obj2str(config)
    const max_len = 20
    if(name.length>max_len){
        document.getElementById('layerName').innerText = name.slice(0, max_len-1)+"..."
    }else{
        document.getElementById('layerName').innerText = name
    }
    
    document.getElementById('layerName').title = name
    
    document.getElementById('layerConfig').innerHTML = config
}
//draw
const draw = (json) => {
    d3.select('.shiftMargin').remove() //remove previous graph
    d3.select('.miniMap').remove() //remove previous graph


    let mode;
    let margin = 10
    if (json.node) {
        info = json.node
        mode = "IR"
    } else {
        info = json.config.layers
        mode = "keras"
    }
    dag = getDag(info, mode, margin)
    let nodes = dag.nodes
    let edges = dag.edges
    let height = dag.height
    let width = dag.width
    
    // let scale = Math.min(window.innerWidth*0.85 / width, (window.innerHeight-85)/height)
    let scale = 1
    let x_shift = 0
    let y_shift = 0
    let svg_w = Math.max(width, window.innerWidth)
    let svg_h = Math.max(height, (window.innerHeight-85))

    let svg = d3.select('#draw')
        .attr("width", svg_w)
        .attr("height", svg_h)

    let def = svg.append("defs")
    def.append("marker")
        .attr("id", "arrow")
        .attr("markerWidth", "10 ")
        .attr("markerHeight", "10 ")
        .attr("refX", "5 ")
        .attr("refY", "5 ")
        .attr("orient", "auto")
        .attr("viewBox", "0 0 20 20")
        .append("path")
        .attr("d", "M0,0 L0,10 L5,5 z")
        .attr("class", "arrowHead")
        .style('fill', '#aaa')

    // var filter = def.append("filter")
    //     .attr("id", "dropshadow")

    // filter.append("feGaussianBlur")
    //     .attr("in", "SourceAlpha")
    //     .attr("stdDeviation", 4)
    //     .attr("result", "blur");
    // // filter.append("feOffset")
    // //     .attr("in", "blur")
    // //     .attr("dx", 2)
    // //     .attr("dy", 2)
    // //     .attr("result", "offsetBlur");

    // var feMerge = filter.append("feMerge");

    // feMerge.append("feMergeNode")
    //     .attr("in", "offsetBlur")
    // feMerge.append("feMergeNode")
    //     .attr("in", "SourceGraphic");

    let shiftMargin = svg.append('g')
    .attr('class', 'shiftMargin')
    .attr("transform", `translate(${window.innerWidth*(0.15) }, 0) scale(1)`)

    let g = shiftMargin.append('g')
        .attr('class', 'scene')
        .attr("transform", `translate(0, 0) scale(1)`)
    
    let g2 = g.append('g')
        .attr("class", "graph")
        .attr('transform', `translate(${x_shift}, ${y_shift}) scale(${scale})`)

    buildGraph(g2, nodes, edges)

    let nodeMasks = d3.selectAll('.node')
        .append("rect")
        .attr('class', 'nodeMask')
        .attr("width", d => 1.2*d.width)
        .attr("height", 1.2*nodeH)
        // .attr('rx', nodeH / 5)
        // .attr('ry', nodeH / 5)
        .attr("transform", d => { return `translate( ${-d.width * 0.6},${-nodeH * 0.6})` })
        .style("fill", "transparent")
        .style("stroke", "none")
        .on("mousedown", function (d) {
            d3.event.preventDefault()
            d3.event.stopPropagation()
            d3.selectAll('.nodeMask')
                .style('stroke', "none")

            d3.select(this)
                .style("stroke", "red")
                .style("stroke-width", 5)
            selectLayer(d, info, mode)
        })
    
    let miniScale = Math.min(miniW / width, miniH / height)
    if(height/width > miniH/miniW ||true){
        miniMap(nodes, edges, width, height, miniScale)
    }

    svg.call(d3.zoom().on('zoom', pan))
    .on("wheel.zoom", scroll)
    .on("dblclick.zoom", null)
    // svg.on('keydown', ()=>{console.info('ddd')})
    // .on('mouseover', ()=>{console.info('mouse over')})
    let shiftDown = false

   d3.select("body")
   .on("keydown", keyZoom)
   .on("keyup", ()=>{shiftDown=false})
    
    function pan(e) {
        let { movementX:x_, movementY:y_ } = d3.event.sourceEvent
        let { k, x, y } = transformParser(d3.select('.scene').attr('transform'))
        
        // k = parseFloat(k)*parseFloat(k_)
        x = parseFloat(x_) + parseFloat(x)
        y = parseFloat(y_) + parseFloat(y)
        // limit x, y
        x = Math.max(-width*k*0.3, Math.min(x, width*k*0.7))
        y= Math.min(0.4 * window.innerHeight, Math.max(-height*k + 0.4 * window.innerHeight, y))
        g.attr('transform', `translate(${x}, ${y}) scale(${k})`);

        d3.select(".mapMask")
            .attr("y", (-y) / k *miniScale)
            .attr("height", miniH*miniScale / k)

        // // a trick to make text svg transform in MS Edge
        // d3.selectAll(".labels").classed("tempclass", true);
        // setTimeout(function () { d3.selectAll(".labels").classed("tempclass", false); }, 40);
    }
    function scroll(e){
        let { k, x, y } = transformParser(d3.select('.scene').attr('transform'))
        if(shiftDown){
            k = d3.event.wheelDeltaY>0?k*1.1:k*0.9
        }else{
            y = parseInt(y) + parseInt(d3.event.wheelDeltaY)
        }

        y= Math.min(0.4 * window.innerHeight, Math.max(-height*k + 0.4*window.innerHeight, y))
        g.attr('transform', `translate(${x}, ${y}) scale(${k})`);
        d3.select(".mapMask")
        .attr("y", (-y) / k *miniScale)
        .attr("height", miniH*miniScale / k)
        
    }
    function keyZoom(e){
        if(d3.event.keyCode==16 ){
            //when shift is down
            shiftDown = true
            return shiftDown
        }else{
            let code = d3.event.keyCode
            let { k, x, y } = transformParser(d3.select('.scene').attr('transform'))
            if(code==65){//if enter "a", zoom in
                k = k*1.1
            }else if(code == 83){//if enter "s", zoom out
                k = k*0.9
            }
            g.attr('transform', `translate(${x}, ${y}) scale(${k })`);
            d3.select(".mapMask")
            .attr("y", (-y) / k *miniScale)
            .attr("height", miniH*miniScale / k)
        }
        }
}

const miniMap = (nodes, edges, width, height, miniScale) => {
    map = d3.select('#draw')
        .append("g")
        .attr("class", "miniMap")
        .attr("transform", `translate(${window.innerWidth * 0.88},0)`)

    let border = map.append("rect")
        .attr('class', 'miniMap_bg')
        .attr("width", window.innerWidth*0.12)
        .attr("height", (window.innerHeight-85)*0.98)
        .attr('rx', 2)
        .attr('ry', 2)
        .style("stroke", "rgba(0, 0, 0, 0.4)")
        .style("fill", "white")

    let mapScene = map.append('g')
        .attr("transform", `translate(${window.innerWidth * 0.06 - width*miniScale/2},0) scale(${miniScale})`)
    buildGraph(mapScene, nodes, edges, label=false)

    let mask = map.append("rect")
        .attr("class", "mapMask")
        .attr("width", "12vw")
        .attr("height", miniScale*miniH)
        .style("fill", "#777")
        .style("opacity", 0.12)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragging)
            .on("end", dragended));

    function dragstarted(d) {
        d3.select(this).raise().classed("dragging", true);
    }

    function dragging(d) {
        let { height, y: mask_y } = d3.select(this).node().getBBox()
        let r = height / (0.98*(window.innerHeight-85))
        let { k, x, y } = transformParser(d3.select('.scene').attr('transform'))
        d3.select(this).attr("y", mask_y + d3.event.dy);
        d3.select('.scene').attr('transform', `translate(${x}, ${y - d3.event.dy / (r)}) scale(${k})`)
    }

    function dragended(d) {
        d3.select(this).classed("dragging", false);
    }

    
}

//  ulti function
function transformParser(string) {
    let translate = string.split('translate(')[1].split(')')[0]
    let k = string.split('scale(')[1].split(')')[0]
    let [x, y] = translate.split(',')
    return { k, x, y }
}

const buildGraph = (g2, nodes, edges, label=true) => {

    let drawNode = g2.selectAll(".node")
        .data(nodes)
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", d => { return `translate(${d.x}, ${d.y})` })

    let layers = drawNode.append("rect")
        .attr('class', 'layers')
        .attr("width", d => (d.width))
        .attr("height", nodeH)
        // .attr('rx', nodeH / 5)
        // .attr('ry', nodeH / 5)
        .attr("transform", d => { return `translate( ${-d.width * 0.5},${-nodeH * 0.5})` })
        .style("fill", d => {
            let op = d.label.split(':').pop()
            return getColor(op)
        })
        .style("stroke", "none")
        .style('opacity', '0.7')
        .classed('shadow', true)
    if(label){
        let labels = drawNode.append("text")
        .attr('class', 'labels')
        .style("text-anchor", "middle")
        .text(d => d.label.split("/").pop())
    }
    
    let drawLink = g2.selectAll(".link")
        .data(edges)
        .enter()
        .append("g")
        .attr("class", "link")

    let links = drawLink.append('path')
        .attr("d", d => arrows(d))
        .attr("stroke", "#aaa")
        .attr("stroke-width", 2)
        .attr('class', 'links')
        .attr("fill", "none")
        .attr("marker-end", "url(#arrow)")
}

const getColor = (op) => {
    let i = OPs.indexOf(op)
    if (i == -1) {
        OPs.push(op)
        return COLORs[(OPs.length - 1) % COLORs.length]
    } else {
        return COLORs[i % COLORs.length]
    }
}
const obj2str = (obj, i = 0) => {
    br = ''
    arr = Object.keys(obj).map(k => {
        v = obj[k]
        space = Array(i).fill('&nbsp&nbsp').join('')
        if(i==0){
            br="<br/>"
        }
        // space = ''
        if (v == null) {
            return `${br}<p>${space}<b>${k}</b>: null </p>`
        } else if (typeof (v) == "object" && Array.isArray(v)) {
            if (typeof (v[0]) == "string"||typeof (v[0]) == "number") {
                return `${br}<p>${space}<b>${k}</b>: ${v.join(',')} </p>`
            } else {
                v = v.map(d => obj2str(d, i + 1))
                return `${br}<p>${space}<b>${k}</b>: <br/>${v.join(' ')} </p>`
            }
        } else if (typeof (v) == "object") {
            return `${br}<p>${space}<b>${k}</b>:<br/> ${obj2str(v, i + 1)} </p>`
        } else {
            return `${br}<p>${space}<b>${k}</b>: ${v} </p>`
        }
    })
    return arr.join('')
}