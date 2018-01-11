import React, { Component } from "react";
import ReactEcharts from "echarts-for-react";
import weights from '../../assets/features/mnist/conv2d_1_w_swap';
import dark from "../../style/dark";
export default class FeatureMap extends Component {
    constructor(props) {
        super(props)
    }
    getOption() {
        const width = 15,
            height = 10,
            margin = 5
        let rangeData = [...Array(weights.length).keys()]
        let grid = rangeData.map(i => {
            return {
                left: `${margin + (width + margin) * (i % 4)}%`,
                top: `${margin / 2 + (height + margin / 2) * Math.floor(i / 4)}%`,
                width: `${width}%`,
                height: `${height}%`
            }
        })
        let axes = rangeData.map(i => {
            return {
                type: 'category',
                gridIndex: `${i}`,
                data: [0, 1, 2],
                show: false
            }
        })
        let series = rangeData.map(i => {
            return {
                name: 'feature' + i,
                type: 'heatmap',
                data: weights[i],
                xAxisIndex: i,
                yAxisIndex: i,
                itemStyle: {
                    emphasis: {
                        borderColor: '#333',
                        borderWidth: 1
                    }
                },
                progressive: 1000,
                animation: false
            }
        })
        let option = {
            tooltip: {},
            grid,
            xAxis: axes,
            yAxis: axes,
            visualMap: {
                type: 'piecewise',
                right: `${margin}%`,
                text: ['pos', 'neg'],
                showLabel: false,
                min: -0.5,
                max: 0.5,
                calculable: true,
                realtime: false,
                splitNumber: 8,
                inRange: {
                    color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
                }
            },
            series,
        }
        return option
    }
    render() {
        return <div className='feature wrapper' 
        style={{overflowY:scroll}}>
            <ReactEcharts
                option={this.getOption()}
                style={{ height: `${window.innerHeight - 40}px` }}
                theme='dark'
            />
        </div>
    }
}