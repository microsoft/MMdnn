export interface StoreState {
    model:Model;
    selectedLayer: string|undefined;
}

export interface Model {
    node:Array<IRNode>
}

export interface IRNode {
    [key:string]:any
}
