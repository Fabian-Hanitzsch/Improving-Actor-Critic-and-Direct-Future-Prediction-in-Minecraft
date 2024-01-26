const THREE = require('three')
const TWEEN = require('@tweenjs/tween.js')

const Entity = require('./entity/Entity')
const minecraftData  = require('minecraft-data')
const { dispose3 } = require('./dispose')
var Canvas = require('canvas');

//const mcData = minecraftData('1.19')
function getEntityMesh (entity, item_count, scene) {
  if (entity.name) {
    try {
      let entity_type = ""
      if (entity.name.toLowerCase() === "item"){
        entity_type = entity.itemid
      }
      else entity_type = entity.name

      const e = new Entity('1.19.1', entity_type, item_count, scene, entity.objectData)

      if (entity.username !== undefined) {

        var Image = Canvas.Image;
        var canvas = Canvas.createCanvas(500, 100);
        //const canvas = document.createElement('canvas')
        //canvas.width = 500
        //canvas.height = 100

        const ctx = canvas.getContext('2d')
        ctx.font = '50pt Arial'
        ctx.fillStyle = '#000000'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'top'

        const txt = entity.username
        ctx.fillText(txt, 100, 0)

        const tex = new THREE.Texture(canvas)
        tex.needsUpdate = true
        const spriteMat = new THREE.SpriteMaterial({ map: tex })
        const sprite = new THREE.Sprite(spriteMat)
        sprite.position.y += entity.height + 0.6

        e.mesh.add(sprite)
      }
      return e.mesh
    } catch (err) {
      console.log(err)
    }
  }

  const geometry = new THREE.BoxGeometry(entity.width, entity.height, entity.width)
  geometry.translate(0, entity.height / 2, 0)
  const material = new THREE.MeshBasicMaterial({ color: 0xff00ff })
  const cube = new THREE.Mesh(geometry, material)
  return cube
}
// 1-1->1, 2-16->2, 17-32->3, 33-48->4, 49-64->5
function stack_count_to_visual_count(item_count) {
  if (item_count == 1) return 1
  if (item_count >= 2 &&(item_count < 17)) return 2
  if (item_count >= 17 &&(item_count < 33)) return 3
  if (item_count >= 33 &&(item_count < 49)) return 4
  return 5
}

class Entities {
  constructor (scene) {
    this.scene = scene
    this.entities = {}
    this.items_count_visual = {}
  }

  clear () {
    for (const mesh of Object.values(this.entities)) {
      this.scene.remove(mesh)
      dispose3(mesh)
    }
    this.entities = {}
  }



  update (entity) {
    if (entity.username === "AgentRL") return

    if (!this.entities[entity.id]) {
      let item_visual_count = 0
      if (entity.name === "item"){
        const item_count = entity.itemcount
        item_visual_count = stack_count_to_visual_count(item_count)
        this.items_count_visual[entity.id] = item_visual_count
        }
      const mesh = getEntityMesh(entity, item_visual_count, this.scene)
      if (!mesh) return
      this.entities[entity.id] = mesh
      this.scene.add(mesh)
    }

    let e = this.entities[entity.id]
    if (entity.name === "item"){
      const item_count = entity.itemcount
      const item_visual_count = stack_count_to_visual_count(item_count)
      if (item_visual_count !== this.items_count_visual[entity.id]){
        this.scene.remove(e)
        dispose3(e)
        this.items_count_visual[entity.id] = item_visual_count
        const mesh = getEntityMesh(entity, item_visual_count, this.scene)
        this.scene.add(mesh)
        e = mesh
        this.entities[entity.id] = e
      }
    }


    if (entity.delete) {
      if (entity.name === "item") delete this.items_count_visual[entity.id]
      this.scene.remove(e)
      dispose3(e)
      delete this.entities[entity.id]
    }

    if (entity.pos) {
      new TWEEN.Tween(e.position).to({ x: entity.pos.x, y: entity.pos.y, z: entity.pos.z }, 50).start()
    }
    if (entity.yaw) {
      const da = (entity.yaw - e.rotation.y) % (Math.PI * 2)
      const dy = 2 * da % (Math.PI * 2) - da
      new TWEEN.Tween(e.rotation).to({ y: e.rotation.y + dy }, 50).start()
    }
  }
}

module.exports = { Entities }
