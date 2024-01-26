import mineflayer from "mineflayer";
import pkg from "prismarine-viewer-extension";
import fs from "fs";
import {Vec3} from "vec3"
const {headless} = pkg;

let mc_server_connection = JSON.parse(fs.readFileSync('../Configs/minecraft-server-connection.json', 'utf8'));
let image_stream_config = JSON.parse(fs.readFileSync('../Configs/image-stream.json', 'utf8'));
let bot_spawned = false

const bot = mineflayer.createBot({
    host: mc_server_connection.host, // minecraft server ip
    port: mc_server_connection.port,       // only set if you need a port that isn't 25565
    username: mc_server_connection.viewer, // minecraft username
    auth: mc_server_connection.auth, // for offline mode servers, you can set this to 'offline'
    checkTimeoutInterval: 3000 * 1000
    //version: "1.19.1",       // only set if you need a specific version or snapshot (ie: "1.8.9" or "1.16.5"), otherwise it's set automatically
    //logErrors:false
    // password: '12345678'        // set if you want to use password-based auth (may be unreliable)
})

bot.once('spawn', () => {
    bot_spawned = true

    console.log("Bot 2 spawned")

    let target_host = ""
    if (image_stream_config["IPC"]){
        target_host = image_stream_config["ip_addr"].toString()
    }
    else {
        target_host = image_stream_config["ip_addr"].toString() + ":" + image_stream_config["port"].toString()
    }

    headless(bot, {
        IPC: image_stream_config["IPC"],
        viewDistance: image_stream_config["view_distance"],
        output: target_host,
        frames: -1,
        width: image_stream_config["image_width"],
        height: image_stream_config["image_height"],
        milliseconds_between_frames: image_stream_config["milliseconds_between_frames"]
    })

    bot.chat("/gamemode spectator")
    waitForBot().then()
})

async function waitForBot(){
    await sleep(image_stream_config["initial_delay_seconds"] * 1000)
    process.send("ready")
}

function sleep(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

process.on("message", async message => {
    if (message === "pause"){
        waitForBot().then()
    }

    else if (bot_spawned){
        const position = message["position"]
        const yaw = message["yaw"]
        const pitch = message["pitch"]
        const target_position = new Vec3(position.x, position.y, position.z)

    // server snaps bot back if distance is over 10 blocks, so we use the tp command instead in those cases (huge lag spike or bot died)
    if (target_position.distanceSquared(bot.entity.position) > 100) {
      console.log("Player is too far away");
      bot.chat("/tp " + target_position.x + " " + target_position.y + " " + target_position.z)
    } else {
      bot.entity.position = target_position;
    }
    bot.entity.yaw = yaw
    bot.entity.pitch = pitch
    }
})