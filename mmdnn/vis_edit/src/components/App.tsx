
import * as React from 'react';
import "./App.css";
import SiderBar from "../containers/SideBar";
import Graph from "../containers/Graph";

// import { Row, Col} from 'antd';


export interface State {
    width: number
}
class App extends React.Component<any, State>{
    public moveStart: number;
    constructor(pros: any) {
        super(pros)
        this.mouseDown = this.mouseDown.bind(this)
        this.mouseUp = this.mouseUp.bind(this)
        this.resize = this.resize.bind(this)
        this.state = { width: window.innerWidth * 0.15 }
    }
    mouseDown(e: any) {
        e.stopPropagation()
        e.preventDefault()
        document.addEventListener("mousemove", this.resize)
        this.moveStart = e.clientX
    }
    mouseUp() {
        document.removeEventListener("mousemove", this.resize)
    }
    resize(e: any) {
        // console.info(e)
        e.stopPropagation()
        e.preventDefault()
        let { width } = this.state
        this.setState({ width: width + e.clientX - this.moveStart })
        this.moveStart = e.clientX
    }
    render() {
        let { width } = this.state
        return (
            <div className="app" >
                <div className="header" style={{ width: "100vw", height: "70px" }}>Visual DNN</div>
                <SiderBar width={width} />
                <span className="resizer-col" style={{ left: width - 20 }}
                    onMouseDown={this.mouseDown}
                    onMouseUp={this.mouseUp}
                    // onMouseLeave={this.mouseUp}
                />
                <Graph width={window.innerWidth - width} />
            </div>
        );
    }
}

export default App;

// helpers

// function getExclamationMarks(numChars: number) {
//     return Array(numChars + 1).join('!');
// }