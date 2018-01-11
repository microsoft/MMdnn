import React from 'react';
import ReactEcharts from 'echarts-for-react';
import {COLORS} from '../../assets/const'
import echarts from 'echarts';
import dark from "../../style/dark";
import axios from 'axios';
echarts.registerTheme('dark', dark)


export default class Chart extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            height: `${window.innerHeight - 40}px`,
            lossData: [[0, 1]],
            accData: [[0, 0]]
        }
    }
    componentDidMount() {
        window.addEventListener('resize', this.resize.bind(this))
        this.resize()
        // this.tick()
        this.intervalID = window.setInterval(this.tick.bind(this), 1000)
    }
    tick() {
        axios.get('http://localhost:3000/fetch_data')
            .then(res => {
                let lossData = JSON.parse(res.data).map(d => [d.batch, d.loss])
                let accData = JSON.parse(res.data).map(d => [d.batch, d.acc])
                this.setState({ lossData, accData })
            })
    }
    componentWillUnmount(){
        window.clearInterval(this.intervalID)
    }
    resize() {
        let height = `${window.innerHeight - 40}px`
        this.setState({ height })
    }
    getOption() {
        let option = {
            tooltip: {
                trigger: 'axis',
                // formatter:(params, ticket, callback)=>{
                //     return `batch:${params.data[0]}, 
                //     ${params.seriesName}:${params.data[1]}`
                // },
            },
            grid: [
                { left: '10%', top: '5%', width: '80%', height: '40%' },
                { left: '10%', top: '55%', width: '80%', height: '40%' },
            ],
            xAxis: [
                {
                    gridIndex: 0,
                    type: 'value',
                    min: 0,
                    name: 'batch'
                },
                {
                    gridIndex: 1,
                    type: 'value',
                    min: 0,
                    name: 'batch'
                },
            ],
            yAxis: [
                {
                    gridIndex: 0,
                    min: 0,
                    name: 'loss',
                    nameTextStyle: {
                        fontSize: 20
                    }
                },
                {
                    gridIndex: 1,
                    min: 0, max: 1,
                    name: 'accuracy',
                    nameTextStyle: {
                        fontSize: 20
                    }
                }
            ],
            series: [
                {
                    name: 'loss',
                    type: 'line',
                    xAxisIndex: 0,
                    yAxisIndex: 0,
                    data: this.state.lossData,
                    areaStyle: {
                        normal: {
                            color: COLORS[0],
                            opacity: 0.5
                        }
                    }
                },
                {
                    name: 'accuracy',
                    type: 'line',
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: this.state.accData,
                    areaStyle: {
                        normal: {
                            color: COLORS[1],
                            opacity: 0.5
                        }
                    }
                },
            ]
        };
        return option
    }
    render() {
        let { height } = this.state
        return <ReactEcharts option={this.getOption()} style={{ height }} theme='dark' />
    }
}