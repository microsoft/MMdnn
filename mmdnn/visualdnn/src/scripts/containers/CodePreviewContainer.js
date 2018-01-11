import { connect } from 'react-redux'
import { 
    changeCode, 
    runCode 
} from '../actions/code_actions'
import {generateCode} from '../ulti/ulti_code'
import CodePreview from '../components/CodePreview';
import * as _ from 'lodash';

// const { exec } = require('child_process');
// const fs = require('fs');
// const path = require('path');



const mapStateToProps = (state, ownProps) => {
    return {
        code: generateCode(state)
    }
}

const mapDispatchToProps = (dispatch, ownProps) => {
    // console.info(ownProps)
    return {
        // onChangeCode: (code) => {
        //   dispatch(changeCode(code))
        // }
        onRunCode: (code) => {
            ownProps.selectMode("plot")
            dispatch(runCode(code))
        }
    }
}

const CodePreviewContainer = connect(
    mapStateToProps,
    mapDispatchToProps
)(CodePreview)
export default CodePreviewContainer



