import mineflayer from 'mineflayer';
import {Vec3} from 'vec3'
import minecraftData from 'minecraft-data';
import fs from 'fs';
import express from "express";
import cors from "cors";
import net from "net";
import {fork} from 'child_process';

import bodyParser from 'body-parser'
const jsonParser = bodyParser.json()


const mcServerConnection = JSON.parse(fs.readFileSync('../Configs/minecraft-server-connection.json', 'utf8'));
const imageStreamConfig = JSON.parse(fs.readFileSync('../Configs/image-stream.json', 'utf8'));
const environmentMineflayerConnectionConfig = JSON.parse(fs.readFileSync('../Configs/environment-mineflayer-connection.json', 'utf8'));

let failedDiggingCount = 0
let diggingPositions = new Set()

let leftPressActivated = false
let isMiningABlock = false
let stoppedDigging = false
let blockDiggingName = ""
let blockDiggingPosition = new Vec3(0, 0, 0)


let justDied = false
let destroyedBlock = false

let bot1Ready = false
let bot2Ready = false
let programFinished = false
let childProcess
let blocksDestroyed = {}
let currentActionDict = {
                                    "insert": "none",
                                    "take": "none",
                                    "craft": "none",
                                    "equip": "none",
                                    "forward": false,
                                    "back": false,
                                    "left": false,
                                    "right": false,
                                    "sprint": false,
                                    "sneak": false,
                                    "use": false,
                                    "attack": false,
                                    "jump": false,
                                    "camera": [0, 0]
                                }

const armorStart = 5


process.on('uncaughtException', err => {
  console.log(`Uncaught Exception: ${err.message}`)
    console.log(err)
  process.exit(1)
})


// ToDo: complete list with every burnable minecraft item
const fuelItems = new Set(["oak_planks", "spruce_planks", "birch_planks", "jungle_planks",
    "acacia_planks", "dark_oak_planks", "mangrove_planks", "oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log",
"dark_oak_log", "mangrove_log", "chest", "crafting_table", "coal", "charcoal", "wooden_shovel", "wooden_pickaxe", "wooden_axe",
"stick"])
const mcData = minecraftData(mcServerConnection.version)

async function sendBotData(){
    if (!bot1Ready) return
    const message = {"position": bot.entity.position, "yaw": bot.entity.yaw, "pitch": bot.entity.pitch}
    childProcess.send(message)
}

function runProcess() {
    childProcess.on("message", message => {
        if (message === "ready") bot2Ready = true
        else {
            bot2Ready = false
            console.log("VIEW BOT DIED THIS SHOULDNT EVEN BE POSSIBLE")
            childProcess.kill()
            shutDown()
        }

    })
    childProcess.on("error", message => {
        console.log("VIEW BOT CRASHED THIS SHOULDNT EVEN BE POSSIBLE")
        childProcess.kill()
        shutDown()
    })
    childProcess.on("close", message => {
        console.log("CRITICAL ERROR VIEW BOT ended")
        if (!programFinished) {
            shutDown()
        }
    })

}


let client = new net.Socket()
let messageLength = 0
let mdata = Buffer.alloc(0);
startClient()

function startClient(){
    if (imageStreamConfig["IPC"]) client.connect("/share/echo.sock", () => {console.log("CONNECTED")})
    else client.connect(5400,"127.0.0.1", () => {console.log("CONNECTED")})

    client.on('data', readMessage);
    client.on("close", function () {
        console.log("Connection to environment got closed")
    })

    client.on("end" , function () {
        console.log("Connection to environment was ended")
    })

    client.on("error" , function (err) {
        console.log("Connection to environment got an error")
        console.log(err)
    })

}

function readMessage(data){
   if (messageLength === 0){
        mdata = Buffer.alloc(0);
        for (let i = 0; i < 4; i++) {
            messageLength = (messageLength * 256) + data[i];
        }
        mdata = data.subarray(4)
    }
    else{
        mdata = Buffer.concat([mdata, data]);
    }
    if(mdata.length === messageLength) {
        messageLength = 0
        processMessage(mdata)
    }
}

function processMessage(message){
    let messageString = message.toString("utf-8")
    let json = JSON.parse(messageString)
    let type = json["type"]
    let currentTime = Date.now()
    let timeNeeded = currentTime - json["start_time"]
    if (type === "get"){
        sendJson({"state": getState(), "time_needed":timeNeeded})
    }
    else if (type === "action"){
        sendJson({"time_needed":timeNeeded})
        executeAction(json["action"]).then()
    }
}

function sendJson(json){
    let message = JSON.stringify(json)
    const buffer = Buffer.from(message, 'utf8')
    const sizebuff = new Uint8Array(4)
    const view = new DataView(sizebuff.buffer, 0)
    view.setUint32(0, buffer.length, true)
    client.write(sizebuff)
    client.write(buffer)
}

function errorFromBot(err){
    if (err.port){
        console.log("connection to server failed, stopping program in a few seconds")
        sleep(60 * 1000).then(process.exit(1))
        }
    else{
        console.log("Unknown Error")
        console.log(err)
        console.trace()
    }
}

function shutDown(){
    console.log("Bot ended somehow")
    console.trace()
    process.exit(1)
}

function startDigging(){
    isMiningABlock = true
    if (bot.targetDigBlock){
        blockDiggingName = bot.targetDigBlock["name"]
        blockDiggingPosition = bot.targetDigBlock["position"]
    }
    else{
        console.log("The bot started digging without a valid block??")
    }

}


const bot = mineflayer.createBot({
    host: mcServerConnection.host, // minecraft server ip
    port: mcServerConnection.port,       // only set if you need a port that isn't 25565
    username: mcServerConnection.username, // minecraft username
    auth: mcServerConnection.auth, // for offline mode servers, you can set this to 'offline'
    checkTimeoutInterval: 3000 * 1000
    // only set if you need a specific version or snapshot (ie: "1.8.9" or "1.16.5"), otherwise it's set automatically
    //logErrors:false
    // password: '12345678'        // set if you want to use password-based auth (may be unreliable)
})
console.log("created bot")

bot.on("error", errorFromBot)
bot.on("end", shutDown)
bot.on('kicked', console.log)
bot.on('death', resetBot)
bot.on("physicsTick", diggingRoutine)
bot.on("physicsTick", sendBotData)
bot.on("startedDigging", startDigging)

bot.on("diggingFinished", () => {
    const blockPositionHash = blockDiggingPosition.x + blockDiggingPosition.y * 1000 + blockDiggingPosition.z * 1000 * 1000
    if (diggingPositions.has(blockPositionHash)) {
        failedDiggingCount += 1
    }
    else {
        diggingPositions.add(blockPositionHash)
    }

    if (blockDiggingName in blocksDestroyed){
        blocksDestroyed[blockDiggingName] += 1
    }
    else {
        blocksDestroyed[blockDiggingName] = 1
    }
    destroyedBlock = true
    isMiningABlock = false
    stoppedDigging = true
})
bot.once('spawn', () => {
    // Stream frames over tcp to a server listening on port 8089, ends when the application stop
    setBotReady().then()
})


async function setBotReady(){
    childProcess = fork("./bot_viewer.js")
    runProcess()

    bot.setControlState("jump", true) // avoid drowning in case of a bad spawn
    while (!bot2Ready){
        await sleep(1000)
    }
    //bot.setControlState("jump", false)
    await sleep(1000)
    bot1Ready = true

}

async function craftItem(name, amount) {
    const item = bot.registry.itemsByName[name]
    const craftingTableID = bot.registry.blocksByName.crafting_table.id

    const craftingTable = bot.findBlock({
        point: bot.entity.position.offset(0, bot.entity.height, 0),
        matching: craftingTableID,
        maxDistance: 4.5
    })

    if (item) {
        const recipe = bot.recipesFor(item.id, null, 1, craftingTable)[0]
        if (recipe) {
            try {
                await bot.craft(recipe, amount, craftingTable)
            } catch (err) {
                console.log(`error making ${name}`)
            }
        }
    } else {
        console.log(`unknown item: ${name}`)
    }
}

async function resetBot(){
    console.log("Resetting bot")
    bot.clearControlStates()
    bot.entity.yaw = 0
    bot.entity.pitch = 0

    // stopDigging handled by the digging plugin
    //if (isMiningABlock) stopDigging().catch()
    isMiningABlock = false
    stoppedDigging = true
    justDied = true
    bot1Ready = false
    bot2Ready = false
    sendBotData().then()
    childProcess.send("pause")

    bot.setControlState("jump", true) // avoid drowning in case of a bad spawn
    while (!bot2Ready){
        await sleep(1000)
    }

    // I am not sure why, but he falls internally (externally he flies) during the reload. trying to turn it down with startFlying results in crashes :/
    while (bot.entity.position.y < -64){
        console.log(bot.entity.position.y)
        await sleep(1000)
    }
    console.log(bot.entity.position.y)

    bot1Ready = true
    console.log("Finished respawning")

}

function attack() {
    if (bot.currentWindow !== null) return
    const entityToInteract = bot.entityAtCursor()
    if (!entityToInteract) return
    if (entityToInteract === bot.entity) return
    if (entityToInteract.name === "item") return
    bot.attack(entityToInteract);

}

function openContainerType(container) {
    const containerName = container.name.toString()
    if (containerName.includes("furnace")) {
        bot.openFurnace(container).catch(e => {
            console.log("failed opening furnace")
        })

    } else if (containerName.includes("anvil")) {
        bot.openAnvil(container).catch(e => {
            console.log("failed opening anvil")
        })
    } else if (containerName.includes("villager")) {
        bot.openVillager(container).catch(e => {
            console.log("failed opening villager")
        })
    } else if (containerName.includes("enchanting_table")) {
        bot.openEnchantmentTable(container).catch(e => {
            console.log("failed opening enchanting table")
        })
    } else {
        bot.openContainer(container).catch(e => {
            console.log("failed opening container")
        })
    }
}

function openContainer(entityToInteract, blockToInteract) {
    const interactableBlocks = ["chest", "furnace", "barrel", "dispenser", "enchanting_table", "anvil"]
    const interactableEntities = ["villager"]

    // If we got an entity at our cursor we can only interact with this entity
    // And the only entity which has an open option are villagers
    if (entityToInteract) {
        for (let i = 0; i < interactableEntities.length; i++) {
            const interaction = interactableEntities[i]
            const entityName = entityToInteract.name.toString()
            if (entityName.includes(interaction)) {
                openContainerType(entityToInteract)
                return true;
            }
        }
    }
    if (!blockToInteract) {
        return false
    }

    const blockName = blockToInteract.name.toString()
    for (let i = 0; i < interactableBlocks.length; i++) {
        const interaction = interactableBlocks[i]
        if (blockName.includes(interaction)) {
            openContainerType(blockToInteract)
            return true;
        }
    }

    return false
}

function getHighestEntranceTime(start, end, direction) {
    let highestEntranceTime = 0
    for (let i = 0; i < 3; i++) {
        const value = direction[i]

        if (value === 0) continue;
        const distanceLow = (end[i] - start[i]) / value
        const distanceHigh = (end[i] - (start[i] - 1)) / value
        let lowestDistance = 100000000
        if (distanceLow < lowestDistance) {
            lowestDistance = distanceLow
        }
        if (distanceHigh < lowestDistance) {
            lowestDistance = distanceHigh
        }

        if (lowestDistance > highestEntranceTime) {
            highestEntranceTime = lowestDistance
        }
    }
    return highestEntranceTime
}

function tryEquipping() {
    // hand, head, torso, legs, feet, off-hand
    const item = bot.heldItem
    if (item === null) return
    if (item.name.includes("leggings")) {
        bot.equip(item, "legs").catch(e => {
            console.log(e)
        })
    } else if (item.name.includes("boots")) {
        bot.equip(item, "feet").catch(e => {
            console.log(e)
        })
    } else if (item.name.includes("helm")) {
        bot.equip(item, "head").catch(e => {
            console.log(e)
        })
    } else if (item.name.includes("chestplate")) {
        bot.equip(item, "torso").catch(e => {
            console.log(e)
        })
    }
}

function placeBlock(blockToInteract) {
    // There are probably better ways of getting which side of the block we want to place our block
    // but it is already too late for me to fix it....

    // view direction as vector (x,y,z)
    const xzLen = Math.cos(bot.entity.pitch)
    const viewZ = -xzLen * Math.cos(bot.entity.yaw)
    const viewY = Math.sin(bot.entity.pitch)
    const viewX = xzLen * Math.sin(-bot.entity.yaw)

    // Made into array to access by index (maybe possible with vector instead)
    const viewDirection = new Vec3(viewX, viewY, viewZ).toArray()
    const blockPosition = blockToInteract.position.toArray()

    let botPosition = bot.entity.position.toArray()
    botPosition[1] = botPosition[1] + bot.entity.height

    let faceDirection = [0, 0, 0]
    let lowestDistance = 1000000

    // time needed until all coordinates from the start position + view direction are inside the target block
    let highestEntranceTime = getHighestEntranceTime(botPosition, blockPosition, viewDirection)

    // find the lowest time needed that is higher or equal to the highestEntranceTime
    for (let i = 0; i < 3; i++) {
        const value = viewDirection[i]

        if (value === 0) continue;
        const distanceLow = (blockPosition[i] - botPosition[i]) / value
        const distanceHigh = (blockPosition[i] - (botPosition[i] - 1)) / value
        if (distanceLow >= highestEntranceTime && distanceLow < lowestDistance) {
            lowestDistance = distanceLow
            faceDirection = [0, 0, 0]
            faceDirection[i] = -1
        }
        if (distanceHigh >= highestEntranceTime && distanceHigh < lowestDistance) {
            lowestDistance = distanceHigh
            faceDirection = [0, 0, 0]
            faceDirection[i] = 1
        }
    }

    // direction to which the block shall be placed
    const faceDirectionVector = new Vec3(faceDirection[0], faceDirection[1], faceDirection[2])

    // try placing block, eat when it is not possible
    const targetBlock = bot.blockAt(blockToInteract.position.plus(faceDirectionVector))
    if (targetBlock) {
        if (targetBlock.name !== "air") {
            bot.consume().catch(e => {
                tryEquipping()
            })
            return;
        }
    }

    bot.placeBlock(blockToInteract, faceDirectionVector).catch(e => {
        bot.consume().catch(e => {
            tryEquipping()
        })
    })

}

function interactBlock(blockToInteract) {
    // too far away
    if (blockToInteract.position.distanceTo(bot.entity.position.offset(0, bot.entity.height, 0)) > 4.5) {
        bot.consume().catch(e => {
            tryEquipping()
        })
        return;
    }
    // bed
    if (bot.isABed(blockToInteract)) {
        bot.sleep(blockToInteract).catch(err => console.log("failed sleeping"));
        return;
    }

    placeBlock(blockToInteract)

}

function rightClick() {
    const entityToInteract = bot.entityAtCursor()
    const heldItem = bot.heldItem

    if (heldItem) {
        if (heldItem.name === "fishing_rod") {
            bot.fish().catch(e => {
                console.log("failed fishing")
            })
            return;
        }
    }
    const blockToInteract = bot.blockAtCursor();
    if (openContainer(entityToInteract, blockToInteract)) {
        return
    }

    if (entityToInteract) {
        bot.useOn(entityToInteract)
        return;
    }

    if (blockToInteract) {
        interactBlock(blockToInteract)
    }


}

function sleep(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

async function stopDigging(){
    bot.stopDigging()
    isMiningABlock = false
    stoppedDigging = true
}

function shouldDig(blockLooking){
    if (!bot1Ready) return false
    if (!blockLooking) return false
    if (!leftPressActivated) return false

    const enitityLooking = bot.entityAtCursor()
    if (enitityLooking || bot.currentWindow) return false

    if (!bot.canDigBlock(blockLooking)) return false

    let sameBlock = false
    if (blockLooking && bot.targetDigBlock) sameBlock = blockLooking.position.equals(bot.targetDigBlock.position) && blockLooking.name === bot.targetDigBlock.name
    if (!bot.targetDigBlock) sameBlock = true

    // Not completly correct, normally the player starts the other block right away, here the bot waits for one
    // physics tick, but when I tried to do it in the same tick it did not work
    if (!sameBlock) return false

    return true
}

function diggingRoutine() {
    if (!bot1Ready) return

    const start_time = Date.now()
    const blockLooking = bot.blockAtCursor(4.5)
    const shouldDigResult = shouldDig(blockLooking)
    const end_time = Date.now()
    const time_needed = (end_time - start_time)

    if (!shouldDigResult && isMiningABlock){
        stopDigging().catch()
    }
    else if (shouldDigResult && !isMiningABlock){
        // catch and ignore the error
        bot.dig(blockLooking, "ignore").catch(err => {
            isMiningABlock = false
            stoppedDigging = true

            if (!err.message){
                console.log(err)
                console.log("Etwas unerwartetes ging schief")
            }
            else if (err.message === 'Block not in view' || err.message === 'dig was called with an undefined or null block'){
                console.log(err)
                console.log("Etwas ging sehr schief")
            }
        }).then()
    }
}



async function insertItem(itemName) {
    if (itemName === "none") return

    if (bot.registry.itemsByName[itemName] === undefined) {
        console.log("unbekanntes item Name (inserting): " + itemName)
        return
    }
    let playerSlot = findItemSlot(itemName, bot.currentWindow.inventoryStart, bot.currentWindow.inventoryEnd)
    if (playerSlot === -1) {
        return
    }

    if (!bot.currentWindow.type.includes("furnace")) {
        let chestFreeSlot = findItemSlot(null, 0, bot.currentWindow.inventoryStart)
        if (chestFreeSlot === -1) {
            return
        }

        await bot.moveSlotItem(playerSlot, chestFreeSlot).catch(e => {
        })
    } else {
        if (fuelItems.has(itemName) && bot.currentWindow.slots[1] === null) {
            await bot.moveSlotItem(playerSlot, 1).catch()
        } else if (bot.currentWindow.slots[0] === null) {
            await bot.moveSlotItem(playerSlot, 0).catch()
        }


    }
}

async function takeItem(itemName) {
    if (itemName === "none") return

    if (bot.registry.itemsByName[itemName] === undefined) {
        console.log("unbekanntes item Name (takeItem): " + itemName)
        return
    }
    let targetItemSlot = findItemSlot(itemName, 0, bot.currentWindow.inventoryStart)
    if (targetItemSlot === -1) {
        return
    }
    let playerItemSlot = findItemSlot(null, bot.currentWindow.inventoryStart, bot.currentWindow.inventoryEnd)
    if (playerItemSlot === -1) return
    await bot.moveSlotItem(targetItemSlot, playerItemSlot).catch(e => {
    })

}

function shouldChestBeClosed(actionDict) {
    return actionDict["forward"] || actionDict["back"] || actionDict["left"] || actionDict["right"] ||
        actionDict["jump"] || actionDict["sneak"] || actionDict["use"]
}

async function holdItem(itemName){
    if (itemName === "air") {
        await bot.unequip("hand").catch(err => {
            //console.log("failed Un equipping")
            //console.log(err)
        })
    }
    else {
        const item = bot.registry.itemsByName[itemName]
        if (!item){
            console.log("unbekanntes item Name (equipping): " + itemName)

            return
        }
        await bot.equip(item.id, "hand").catch(err =>{
            //console.log("failed equipping")
            //console.log(err)
    })
    }


}

// Agent acts according to the given action
async function executeAction(actionDict) {
    if (!bot1Ready) return
    currentActionDict = actionDict

    if (bot.currentWindow) {
        await insertItem(actionDict["insert"]).catch()
        await takeItem(actionDict["take"])
        if (shouldChestBeClosed(actionDict)) await bot.currentWindow.close()
    }

    if (actionDict["sprint"]) {
        actionDict["sneak"] = false
        actionDict["back"] = false
        actionDict["forward"] = true
    }

    bot.setControlState("forward", actionDict["forward"])
    bot.setControlState("back", actionDict["back"])
    bot.setControlState("left", actionDict["left"])
    bot.setControlState("right", actionDict["right"])
    bot.setControlState("jump", actionDict["jump"])
    bot.setControlState("sneak", actionDict["sneak"])
    bot.setControlState("sprint", actionDict["sprint"])

    const rotationVertical = actionDict["camera"][0]
    const rotationHorizontal = actionDict["camera"][1]
    if (!bot.currentWindow) {
        bot.entity.yaw += rotationHorizontal * Math.PI / 180
        bot.entity.pitch += rotationVertical * Math.PI / 180
        if (bot.entity.pitch > Math.PI/2) bot.entity.pitch = Math.PI/2
        if (bot.entity.pitch < -Math.PI/2) bot.entity.pitch = -Math.PI/2

    }

    if (actionDict["use"]) {
        await rightClick()
    }

    if (actionDict["attack"]) {
        if (!leftPressActivated) {
            await attack()
            leftPressActivated = true
        }
    }
    else{
        leftPressActivated = false
    }

    if (actionDict["equip"] !== "none") {
        await holdItem(actionDict["equip"])
    }

    if (actionDict["craft"] !== "none") {
        const craftName = actionDict["craft"]
        if (bot.registry.itemsByName[craftName] === undefined) {
            console.log("unbekanntes item Name (crafting): " + craftName)
            return
        }
        craftItem(craftName, 1).then()
    }
}

function addSlotToState(slot, state) {
    const name = slot.name
    const itemData = mcData.itemsByName[name]
    const maxDurability = itemData.maxDurability
    if (maxDurability !== undefined) {
        let j = 0
        // inefficient if the bot got many items
        while (j < 100) {
            if (state[name + "#" + j] === undefined) {
                state[name + "#" + j] = {
                    "amount": slot.count,
                    "damage": slot.nbt.value.Damage.value,
                    "max_damage": maxDurability
                }
                break
            }
            j += 1
        }
    } else {
        if (state[name] === undefined) {
            state[name] = {"amount": slot.count, "damage": -1, "max_damage": -1}
        } else {
            state[name]["amount"] += slot.count
        }
    }
    return state
}

function slotToHandState(slot) {
    let state = {}
    if (slot !== null) {
        const name = slot.name
        const amount = slot.count
        const itemData = mcData.itemsByName[name]
        const maxDurability = itemData.maxDurability
        if (maxDurability !== undefined) {
            state = {"type": name, "damage": slot.nbt.value.Damage.value, "max_damage": maxDurability, "amount": amount}
        } else {
            state = {"type": name, "damage": -1, "max_damage": -1, "amount": amount}
        }
    } else {
        state = {"amount": 1, "type": "air", "damage": -1, "max_damage": -1}
    }
    return state
}

function findItemSlot(itemName, start, end){
    let targetItemSlot = -1
    for (let i = start; i < end; i++) {
        const slot = bot.currentWindow.slots[i]
        if (slot === null && itemName === null){
            targetItemSlot = i
            break
        }
        // only works with null, not undefined
        else if (slot?.name === itemName){
            targetItemSlot = i
            break
        }
    }
    return targetItemSlot
}

function canHarvestBlock(){
    if (!bot.targetDigBlock) return false
    else {
        const blockByMcData = mcData.blocks[bot.targetDigBlock.type]
        if ("harvestTools" in blockByMcData){
            if (!bot.heldItem) return false
            if (!(bot.heldItem.type in blockByMcData["harvestTools"])){
                return false
            }
        }
        return true
    }
}



function getState() {
    const receiveTime = Date.now()
    const state = {} // state des Environments

    if (!bot1Ready && !justDied) return state
    if (justDied) {
        state["just_died"] = true
        justDied = false}
    else{
        state["just_died"] = false
    }
    state["destroyed_block"] = destroyedBlock
    destroyedBlock = false
    state["block_digging"] = bot.targetDigBlock
    state["is_in_water"] = bot.entity.isInWater
    state["current_action"] = currentActionDict

    if (bot.targetDigBlock){
        state["expected_dig_time"] = bot.digTime(bot.targetDigBlock)
    }
    else {
       state["expected_dig_time"] = 10000000
    }


    // state["expected_dig_time"] =
    state["failed_digging_count"] = failedDiggingCount
    state["should_dig"] = shouldDig(bot.targetDigBlock)
    state["can_harvest_block"] = canHarvestBlock()
    state["block_name"] = blockDiggingName
    state["stopped_digging"] = stoppedDigging
    stoppedDigging = false


    state["receive_time"] = receiveTime
    state["blocks_destroyed"] = blocksDestroyed

    const miningProgress = bot.getDigProgress()
    const botInventory = bot.inventory
    let inventoryState = {}
    let inventoryAirAmount = 0

    for (let i = botInventory.inventoryStart; i < botInventory.inventoryEnd; i++) {
        const slot = botInventory.slots[i]
        if (slot === null){
            inventoryAirAmount++
            continue
        }
        inventoryState = addSlotToState(slot, inventoryState)
    }
    inventoryState["air"] = {"amount": inventoryAirAmount, "damage": -1, "max_damage": -1}

    let armorState = {}
    armorState["head"] = slotToHandState(botInventory.slots[armorStart])
    armorState["torso"] = slotToHandState(botInventory.slots[armorStart + 1])
    armorState["legs"] = slotToHandState(botInventory.slots[armorStart + 2])
    armorState["feets"] = slotToHandState(botInventory.slots[armorStart + 3])

    let windowState = {}

    if (bot.currentWindow) {
        let windowAirAmount = 0
        for (let i = 0; i < bot.currentWindow.inventoryStart; i++) {
            const slot = bot.currentWindow.slots[i]
            if (slot === null){
                windowAirAmount++
                continue
            }
            windowState = addSlotToState(slot, windowState)
        }
        windowState["air"] = {"amount": windowAirAmount, "damage": -1, "max_damage": -1}
    }

    state["inventory"] = inventoryState
    state["armor"] = armorState
    state["main_hand"] = slotToHandState(bot.heldItem)
    state["second_hand"] = slotToHandState(botInventory.slots[botInventory.inventoryEnd])
    state["health"] = bot.health
    state["hunger"] = bot.food
    state["experience"] = bot.experience.progress
    state["level"] = bot.experience.level
    state["mining_progress"] = miningProgress
    state["window"] = windowState

    return state;
}

/*

width = 0
for (z = -width; z++; z<=width)
x = width
x = -width


x = 0
z = 0

x += 1
z += 1 * 2
x -= 1 * 2
z -= 1 * 2
x += 1 * 3


 */

function validBlock(x,y,z){
    const targetBlock = bot.blockAt(bot.entity.position.offset(x,y,z))
    if (targetBlock && targetBlock["name"] === "grass_block"){
        const upperBlock1 = bot.blockAt(bot.entity.position.offset(x,y+1,z))
        const upperBlock2 = bot.blockAt(bot.entity.position.offset(x,y+2,z))
        if (!upperBlock1 || !upperBlock2){
            return false
        }
        if (!upperBlock1["boundingBox"] || !upperBlock2["boundingBox"]){
            console.log("boundingBox not found")
            console.log("upper Block 1")
            console.log(upperBlock1)
            console.log("upper Block 2")
            console.log(upperBlock2)
            return false
        }

        if (upperBlock1["boundingBox"] === "empty" && upperBlock2["boundingBox"] === "empty"){
            return true
        }
    }
    return false
}

function executePossiblyValidBlock(x, y, z){
    if (validBlock(x,y,z)){
        console.log("new Spawn position found")
        console.log(bot.entity.position.offset(x,y,z))
        bot.chat("/tp " + (Math.floor(bot.entity.position.x) + x + 0.5) + " " + (bot.entity.position.y + y + 1) + " " + (Math.floor(bot.entity.position.z) + z + 0.5))
        return true
    }
    return false
}

function getNewSpawnPosition(){
    console.log("finding fitting spawn position")
    let targetPositionFound = false

    for (let y=10; y>=-10; y--){
        if (targetPositionFound) break
        let width = 0
        while (width <= 10 && !targetPositionFound){
            let changing_number = 0
            while (changing_number <= width && !targetPositionFound){
                if (executePossiblyValidBlock(width,y,changing_number)) targetPositionFound = true
                else if (executePossiblyValidBlock(-width,y,changing_number)) targetPositionFound = true
                else if (executePossiblyValidBlock(changing_number,y,width)) targetPositionFound = true
                else if (executePossiblyValidBlock(changing_number,y,-width)) targetPositionFound = true

                // we check closer blocks first and then move away from the center
                if (changing_number > 0) changing_number = -changing_number
                else changing_number = -changing_number + 1
            }
            width++
        }
    }

}

/**
 * Initialize webserver
 * */
const app = express();
app.use(cors())
const webServerPort = environmentMineflayerConnectionConfig.http_mineflayer_port;
const apiPrefixMineflayer = '/mineflayer'

app.post(apiPrefixMineflayer + '/end', jsonParser, function (req, res) {
    console.log("System shutting down")
    res.sendStatus(201)
    programFinished = true
    childProcess.kill()
    process.exit(0)
    res.end()

});

app.post(apiPrefixMineflayer + '/reset', jsonParser, function (req, res) {
    console.log("System destroy dropped Blocks")
    bot.chat("/kill @e[type=item]")
    console.log("Clear Inventory")
    bot.chat("/clear")
    if (isMiningABlock){
        stopDigging()
    }
    leftPressActivated = false
    bot.clearControlStates()
    bot.entity.yaw = 0
    bot.entity.pitch = 0
    getNewSpawnPosition()
    sendBotData().then()
    bot.setControlState("jump", true) // same control state as when the episode started
    res.end()
});


app.listen(webServerPort, () => {
console.log('WebServer on port ' + webServerPort);
});
