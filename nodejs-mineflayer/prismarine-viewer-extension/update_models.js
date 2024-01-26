const fs = require('fs-extra')



//copyJSON()
updateJSON()

function copyJSON () {
    const block_models = require('./public/blocksStates/1.19.1.json')
    const target_path = "./public/blocksStates/tmp.json"
    const blocksStates = JSON.stringify(block_models, null, 1)
    fs.writeFileSync(target_path, blocksStates)

}

function updateJSON() {
    const block_models = require("./public/blocksStates/tmp.json")
    const target_path = './public/blocksStates/1.19.1.json'
    const blocksStates = JSON.stringify(block_models)
    fs.writeFileSync(target_path, blocksStates)
}