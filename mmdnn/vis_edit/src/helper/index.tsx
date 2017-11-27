
let names:string[] = []
export function getColor(name:string):string{
    const COLORS: string[]=[
        ' #9e0142',
        ' #fdae61',       
        ' #d53e4f',
        ' #e6f598',
        ' #f46d43',
        ' #ffffbf',
        ' #66c2a5',      
        ' #fee08b',
        ' #3288bd',
        ' #5e4fa2']
    let idx:number = names.indexOf(name)
    let numColor = COLORS.length
    if(idx === -1){
        names.push(name)
        return COLORS[(names.length-1) % numColor]
    }else{
        return COLORS[idx % numColor]
    } 
}