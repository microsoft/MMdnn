import React from "react";
import RightPanel from "./RightPanel";
import GraphContainer from "./GraphContainer";
import ParBoardContainer from "./ParBoardContainer";
import ToolBoxContainer from "./ToolBoxContainer";

export default class App extends React.Component {
  render() {
    return (
      <div className='App'>
        <div className="row">
          <div className="col-sm-1">
          <ToolBoxContainer  />
          </div>
          <div className='col-sm-6'>
            <GraphContainer />
            <ParBoardContainer />
          </div>
          <div className='col-sm-5'>
          <RightPanel />
          </div>
        </div>
      </div>
    );
  }
}


