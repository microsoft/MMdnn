import { getLayerColor, COLORS } from '../assets/def';
export const layerStyle = ( width, height, x=0, y=0,z, name) => {
    // let main=['Conv2D', 'Dense', 'MaxPooling2D']
    // let index = main.findIndex(d=>d==name)
    // let fz = (index==-1)?12:20
    let fz=12
    return {
        // position:'relative',
        zIndex: z,
        height: `${height}px`,
        width: `${width}px`,
        // backgroundColor: getLayerColor(name),
        backgroundColor:COLORS[0],
        aligh: 'center',
        margin: '5px auto',
        padding: '2px',
        textAlign: 'center',
        color: "#333",
        fontSize: `${fz}px`,
        fontFamily: 'sans-serif',
        transform: `translate(${x}px, ${y}px)`,
        borderRadius: '5px',
        boxShadow: '2px 2px 1px #111'
    }
}

export const inOutStyle = (name, x=0, y=0) => {
    return {
        height: `30px`,
        width: `180px`,
        backgroundColor: '#d2f5a6',
        margin: '5px auto',
        padding: '5px',
        textAlign: 'center',
        color: "#333",
        fontSize: '20px',
        fontFamily: 'sans-serif',
        transform: `translate(${x}px, ${y}px)`,
        borderRadius: '50%',
        border:'solid 2px #eee'
    }
}
