## remaing bugs
+ implement select for par board
+ use outlayer id as module id can not handle 1 submodule inside another submodule
+ modify code generator for multiple submodules
+ a module cannot have inputs when copying it
+ can only have a small module and create a bigger one
+ the relocation of selection box after scale and scroll 
## Todo
> short term

+ the lambda layer
+ the attention mechanism
+ reconstruct the store structure and the rendering mechanism


> at least not now
+ type check with parameters(can give a format)
+ when a module/layer selected, highlight the corresponding code

> long term


## In Progress
+ visualize a ready trained model, (use keras h5 file, do some experiments first)
+ svg performance problem (for selection, no need to rerender the whole dag)
+ lambda layer
+ how to viz a complicated RNN

## Done
+ <s>core,merge layers supported</s>
+ <s>module save and reload</s>
+ <s>import external model:IR and keras</s>
+ <s>new code converter</s>
+ <s>zoom by "shift +/-"</s>
+ <s>outputshape(trace it by id)</s>
+ <s>layers dropdown menu</s>
+ <s>copy, paste,delete</s>
+ <s>the calculation needed for each layer</s>
+ <s>better attr from IR model and keras model</s>
+ <s>copy and past layer/modules</s>
+ <s>compress sub modules</s>
+ <s>module inside a module</s>
## about redux structure
now, only store layers in redux store

> state = {
>    
>     layers : array<layer>,
>     dataset: string,
>     compiler： object,
>     run: object, 
>     selected: String (layer id),
>     shape：array<{id:string, shape:array<number>}>
> }

>  interface layer = {    
>
>      name: string,
>      acts: array<act>
>      id: (name:string)_index,
>      inputs: array<id>,
>      folded: boolean,
>
>  }

> interface act = {
>
>      name: string,    
>      pars: object
>
> }



## about weight sharing 
how to do weight sharing:

select two node, right click, select bind

how can I know a node is binded with another one:

show the feature on the Parboard??
(give a name, or a href link)

## about layout
toolbox: col-sm-1， height: 100%
graphview: col-sm-6, height; 75%
parboard: col-sm-6, height;25%
codeview: col-sm-5

