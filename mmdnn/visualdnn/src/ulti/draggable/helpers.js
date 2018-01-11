/*** Helper functions - they are decoupled because of testability */


/**
 * @param {array} items
 * @param {number} indexFrom
 * @param {number} indexTo
 * @returns {array}
 */
export function swapArrayElements(items, indexFrom, indexTo) {
  var item = items[indexTo];
  items[indexTo] = items[indexFrom];
  items[indexFrom] = item;
  return items;
}

/**
 * @param {array} items
 * @param {number} indexFrom
 * @param {number} indexTo
 * @returns {array}
 */
export function reorderArrayElements(items, indexFrom, indexTo) {
  if(indexFrom<indexTo){
      items.forEach((d, i)=>{
          d=(indexFrom<=i<indexTo)?items[i+1]:d
      }, this);
  }else{
      items.forEach((d, i)=>{
          d=(indexFrom>=i>indexTo)?items[i-1]:d
      }, this);
  }
  items[indexTo]=items[indexFrom]
  return items
}

/**
 * @param {number} mousePos
 * @param {number} elementPos
 * @param {number} elementSize
 * @returns {boolean}
 */
export function isMouseBeyond(mousePos, elementPos, elementSize, moveInMiddle) {
  var breakPoint;
  if(moveInMiddle){
    breakPoint = elementSize / 2; //break point is set to the middle line of element
  }else{
    breakPoint = 0
  }
  var mouseOverlap = mousePos - elementPos;
  return mouseOverlap > breakPoint;
}