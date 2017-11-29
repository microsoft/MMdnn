import * as React from "react";
import { Model } from "../types"
import "./SideBar.css"
import { Menu, Icon, Button } from 'antd';
import { ClickParam } from 'antd/lib/menu';
import { IRNode } from "../types";
const SubMenu = Menu.SubMenu;
// const MenuItemGroup = Menu.ItemGroup;
export interface Props {
    nodes: IRNode[];
    width:number;
    onImportModel: (json: Model) => void
    onSelectLayer: (name:string) => void
}
export interface State {
    searchKey: string,
    // width:number
}
class SiderBar extends React.Component<Props, State> {
    public file: any; moveStart:number;
    constructor(pros: any) {
        super(pros)
        this.onImportModel = this.onImportModel.bind(this)
        // this.mouseDown = this.mouseDown.bind(this)
        // this.mouseUp = this.mouseUp.bind(this)
        // this.resize = this.resize.bind(this)
        this.state = { searchKey: ""}
    }
    handleClick = (e: ClickParam) => {
        switch (e.keyPath[1]) {
            case "all_layers":
                if(e.key!="search"){
                    this.props.onSelectLayer(e.key)
                }
            case "add_layers":
                break;
            default:
                break;
        }
    }
    onImportModel() {
        let file: any = this.file.files[0]
        let reader = new FileReader()
        reader.onload = (e: any) => {
            this.props.onImportModel(JSON.parse(e.target.result))
            // console.info(JSON.parse(e.target.result))
        }
        reader.readAsText(file)
    }
    search(e: any) {
        let searchKey = e.target.value.toLowerCase()
        // console.info(e.target.value)
        this.setState({ searchKey })
    }
    
    render() {
        let { nodes } = this.props
        let {searchKey} =  this.state
        // console.info(width)
        return (
            <div className="sideBar" style={{width:this.props.width}}>
                <span className="menuItem ant-menu-item ">
                    <div className="fileinputs">
                        <input type="file"
                            className="file"
                            ref={(ref) => this.file = ref}
                            onChange={this.onImportModel} />

                        <div className="fakefile">
                            {/* <input type="button" value="Import Model" /> */}
                            <Button className="inputButton ">
                                <span className="menuItem"><Icon type="folder-open" />Import Model</span>
                            </Button>
                        </div>
                    </div>
                </span>
                <Menu
                    // theme="dark"
                    onClick={this.handleClick}
                    defaultSelectedKeys={[]}
                    defaultOpenKeys={['all_layers', 'conv_layers']}
                    mode="inline"
                >
                    <SubMenu key="all_layers" className="allLayers" title={<span className="menuItem"><Icon type="barcode" />All Layers</span>}>
                        {(nodes.length == 0) ?
                            <Menu.Item key="search">No layer yet</Menu.Item> :
                            <Menu.Item key="search" className="search">
                                <Icon type="search" />
                                <input type="text" id="myInput"
                                    onChange={this.search.bind(this)}
                                    placeholder="Search for layers.."
                                    title="Type in a name"
                                /></Menu.Item>
                        }

                        {nodes.filter((node)=>node.name.toLowerCase().indexOf(searchKey)>-1).map(node => {
                            return <Menu.Item key={node.name}>{node.name.split('/').reverse().join('/')}</Menu.Item>
                        })}
                    </SubMenu>
                    <SubMenu key="add_layers" title={<span className="menuItem"><Icon type="barcode" />Add Layer</span>}>
                        <SubMenu key="core_layers" title={<span className="layerClass">Core Layers</span>}>
                            <Menu.Item key="1"><span className="layerName">Layer 1</span></Menu.Item>
                            <Menu.Item key="2">Layer 2</Menu.Item>
                        </SubMenu>
                        <SubMenu key="conv_layers" title={<span className="layerClass">Conv Layers</span>}>
                            <Menu.Item key="3">Layer 3</Menu.Item>
                            <Menu.Item key="4">Layer 4</Menu.Item>
                        </SubMenu>
                    </SubMenu>
                </Menu>
            </div>
        );
    }
}

export default SiderBar