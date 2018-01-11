import echarts from 'echarts';
import EchartsMap from 'echarts-map'
import React, { Component } from 'react';
import elementResizeEvent from 'element-resize-event';
// import {COLORS} from 'constants';
// import { GRADIENT_COLORS as colors } from 'constants';
import _ from 'lodash';


const zrender = echarts.zrender;
const graphic = echarts.graphic;

const ShapeCircle = graphic.Circle;
const ShapeLine = graphic.Line;
const ShapeText = graphic.Text;
const ShapeImage = graphic.Image;
const ShapePath = graphic.Path;
const ShapeRect = graphic.Rect;
const ShapeGroup = graphic.Group;

// const ShapeGroup = {};
const ShapeGroupUtil = {
    generate: function generate({ left = 0, top = 0, children, ...props }) {
        let g = new ShapeGroup();
        g.position[0] = left;
        g.position[1] = top;
        Object.assign(g, props);
        [].concat(_.flatten(children)).map(shape => {
            let Shape = shape.type();
            if (Shape === ShapeGroup) {
                g.add(generate(shape.props));
            }
            else {
                g.add(new Shape(shape.props));
            }
        });
        return g;
    }
};

let theme = { color: [
    //   "#fdae61", 
    //   "#fee090", 
    //   "#e0f3f8", 
    //   "#abd9e9", 
    //   "#74add1", 
    //   "#4575b4"

    // '#FFF59D',
    // '#FFBF54',

    // '#B2EBF2',
    // '#26C6DA',
    // '#00B8D4',

    // '#90CAF9',
    // '#42A5F5',
    // '#1976D2'

    '#e8ae4d',
    '#4fbaa1',
    '#507ed4',
    '#FFE082',
    '#B2EBF2',
    '#90CAF9',
    '#26C6DA',
    


    ], textStyle: { color: '#fff' }
};
[
    'title',
    'legend',
    'dataZoom',
    'visualMap',
    'tooltip',
].map(k => _.set(theme, [k], {
    textStyle: {
        color: '#fff'
    }
}));

echarts.registerTheme('dark', theme);
export const colors = theme.color
export default class ReactEcharts extends Component {
    constructor(props) {
        super(props);
    }
    // first add
    componentDidMount() {
        const props = this.props;
        let echartObj = this.renderEchartDom();
        let onEvents = {};
        Object.keys(props).map(key => {
            let mat = key.match(/^on([A-Z]\w+)$/);
            if (mat) {
                onEvents[mat[1].toLowerCase()] = props[key];
            }
        });

        this.bindEvents(echartObj, onEvents);
        // on chart ready
        if (typeof this.props.onChartReady === 'function') this.props.onChartReady(echartObj);

        // on resize
        elementResizeEvent(this.refs.echartsDom, function () {
            echartObj.resize();
        });
    }
    // update
    componentDidUpdate() {
        this.renderEchartDom();
        this.bindEvents(this.getEchartsInstance(), this.props.onEvents || []);
    }
    // remove
    componentWillUnmount() {
        echarts.dispose(this.refs.echartsDom)
    }

    //bind the events
    bindEvents(instance, events) {
        var _loop = function _loop(eventName) {
            // ignore the event config which not satisfy
            if (typeof eventName === 'string' && typeof events[eventName] === 'function') {
                // binding event
                instance.off(eventName);
                instance.on(eventName, function (param) {
                    events[eventName](param, instance);
                });
            }
        };

        for (var eventName in events) {
            _loop(eventName);
        }

    }
    // render the dom
    renderEchartDom() {
        // init the echart object
        let echartObj = this.getEchartsInstance();
        // set the echart option
        echartObj.setOption(this.props.option, this.props.notMerge || false, this.props.lazyUpdate || false);
        // set loading mask
        if (this.props.showLoading) echartObj.showLoading();
        else echartObj.hideLoading();

        return echartObj;
    }
    getEchartsInstance() {
        const { echartsDom } = this.refs;
        const { theme } = this.props;
        return echarts.getInstanceByDom(echartsDom) || echarts.init(echartsDom, theme || 'dark');
    }
    render() {
        let style = this.props.style || {
            height: 300
        };
        // for render
        return (
            <div ref='echartsDom'
                className={this.props.className}
                style={style} />
        );
    }
}

export const ZRender = class Render extends ReactEcharts {
    _exec = (zr) => {
        const { exec, children } = this.props;
        if (exec) {
            exec.call(zr);
        }
        else if (children) {
            let chd = _.flattenDeep([].concat(children));
            zr.clear();
            chd.map(shape => {
                if (!shape) {
                    return;
                }
                let Shape = shape.type();
                if (Shape === ShapeGroup) {
                    zr.add(ShapeGroupUtil.generate(shape.props));
                }
                else {
                    zr.add(new Shape(shape.props));
                }
            });
        }
    };
    renderEchartDom() {
        // init the echart object
        let echartObj = this.getEchartsInstance();
        this._exec(echartObj);
        return echartObj;
    }
    getEchartsInstance() {
        let t = this;
        const { echartsDom } = t.refs;

        let echartObj = zrender.getInstance(this.id);
        if (!echartObj) {
            echartObj = zrender.init(echartsDom);

            this.id = echartObj.id;
            var resize = echartObj.resize;
            echartObj.resize = function () {
                resize.call(echartObj);
                t._exec(echartObj);
            };
        }
        return echartObj;
    }
    componentWillUnmount() {
        this.getEchartsInstance().dispose();
    }
}

ZRender.Circle = () => ShapeCircle;
ZRender.Line = () => ShapeLine;
ZRender.Text = () => ShapeText;
ZRender.Image = () => ShapeImage;
ZRender.Path = () => ShapePath;
ZRender.Rect = () => ShapeRect;
ZRender.Group = () => ShapeGroup;