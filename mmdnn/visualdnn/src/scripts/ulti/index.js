export * from './ulti_code'
export * from './ulti_layer'
export const js2py = (obj) => {
    switch (typeof (obj)) {
        case ('string'):
            switch (obj) {
                case ('true' || 'True'):
                    return 'True'
                case ('false' || 'False'):
                    return 'False'
                case ('none' || 'None'):
                    return 'None'
                default:
                  return `"${obj}"`
            }
        case ('boolean'):
           return obj?`True`:`False`
        case('object'):
           if(obj==null){
               return `None`
           }else{
               return js2py(obj)
           }
        default:
           return obj

    }
}