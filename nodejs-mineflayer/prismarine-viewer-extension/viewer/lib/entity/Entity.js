/* global THREE */

const entities = require('./entities.json')
const item_names = require('../../../public/textures/1.19.1/items_textures.json')
const { loadTexture } = globalThis.isElectron ? require('../utils.electron.js') : require('../utils')

const minecraftData = require('minecraft-data')
const mcData = minecraftData('1.19')
const blocks_with_ids = mcData.blocksByStateId

const full_block = {
          "bones": [
          {
            "name": "bottom",
            "pivot": [0, 6, 0],
            "cubes": [
                {
                "origin": [-8, -8, -8],
                "size": [16, 16, 16],
                "rotation": [0, 0, 0],
                "uv": [0, 0]
              }
              ]
          },
        ],
        "texturewidth": 16,
        "textureheight": 16
      }


const dropped_block = {
        "bones": [
          {
            "name": "bottom",
            "pivot": [0, 6, 0],
            "cubes": []
          },
        ],
        "texturewidth": 4,
        "textureheight": 4
      }

const block_1 = {
                "origin": [0, 2, 0],
                "size": [4, 4, 4],
                "rotation": [0, 0, 0],
                "uv": [0, 0]
              }
const block_2 = {
                "origin": [2, 4, 2],
                "size": [4, 4, 4],
                "rotation": [0, 0, 0],
                "uv": [0, 0]
              }
const block_3 = {
                "origin": [-2, 0, -2],
                "size": [4, 4, 4],
                "rotation": [0, 0, 0],
                "uv": [0, 0]
              }
const block_4 = {
                "origin": [-2, 2.5, 0.5],
                "size": [4, 4, 4],
                "rotation": [0, 0, 0],
                "uv": [0, 0]
              }
const block_5 = {
                "origin": [-3, 4, 2],
                "size": [4, 4, 4],
                "rotation": [0, 0, 0],
                "uv": [0, 0]
              }
const block_stack_list = [block_1, block_2, block_3, block_4, block_5]

const elemFaces = {
  up: {
    dir: [0, 1, 0],
    u0: [0, 0, 1],
    v0: [0, 0, 0],
    u1: [1, 0, 1],
    v1: [0, 0, 1],
    corners: [
      [0, 1, 1, 0, 0],
      [1, 1, 1, 1, 0],
      [0, 1, 0, 0, 1],
      [1, 1, 0, 1, 1]
    ]
  },
  down: {
    dir: [0, -1, 0],
    u0: [1, 0, 1],
    v0: [0, 0, 0],
    u1: [2, 0, 1],
    v1: [0, 0, 1],
    corners: [
      [1, 0, 1, 0, 0],
      [0, 0, 1, 1, 0],
      [1, 0, 0, 0, 1],
      [0, 0, 0, 1, 1]
    ]
  },
  east: {
    dir: [1, 0, 0],
    u0: [0, 0, 0],
    v0: [0, 0, 1],
    u1: [0, 0, 1],
    v1: [0, 1, 1],
    corners: [
      [1, 1, 1, 0, 0],
      [1, 0, 1, 0, 1],
      [1, 1, 0, 1, 0],
      [1, 0, 0, 1, 1]
    ]
  },
  west: {
    dir: [-1, 0, 0],
    u0: [1, 0, 1],
    v0: [0, 0, 1],
    u1: [1, 0, 2],
    v1: [0, 1, 1],
    corners: [
      [0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1],
      [0, 1, 1, 1, 0],
      [0, 0, 1, 1, 1]
    ]
  },
  north: {
    dir: [0, 0, -1],
    u0: [0, 0, 1],
    v0: [0, 0, 1],
    u1: [1, 0, 1],
    v1: [0, 1, 1],
    corners: [
      [1, 0, 0, 0, 1],
      [0, 0, 0, 1, 1],
      [1, 1, 0, 0, 0],
      [0, 1, 0, 1, 0]
    ]
  },
  south: {
    dir: [0, 0, 1],
    u0: [1, 0, 2],
    v0: [0, 0, 1],
    u1: [2, 0, 2],
    v1: [0, 1, 1],
    corners: [
      [0, 0, 1, 0, 1],
      [1, 0, 1, 1, 1],
      [0, 1, 1, 0, 0],
      [1, 1, 1, 1, 0]
    ]
  }
}

function dot (a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function addCube (attr, boneId, bone, cube, texWidth = 64, texHeight = 64) {
  const cubeRotation = new THREE.Euler(0, 0, 0)
  if (cube.rotation) {
    cubeRotation.x = -cube.rotation[0] * Math.PI / 180
    cubeRotation.y = -cube.rotation[1] * Math.PI / 180
    cubeRotation.z = -cube.rotation[2] * Math.PI / 180
  }
  for (const { dir, corners, u0, v0, u1, v1 } of Object.values(elemFaces)) {
    const ndx = Math.floor(attr.positions.length / 3)

    for (const pos of corners) {
      const u = (cube.uv[0] + dot(pos[3] ? u1 : u0, cube.size)) / texWidth
      const v = (cube.uv[1] + dot(pos[4] ? v1 : v0, cube.size)) / texHeight

      const inflate = cube.inflate ? cube.inflate : 0
      let vecPos = new THREE.Vector3(
        cube.origin[0] + pos[0] * cube.size[0] + (pos[0] ? inflate : -inflate),
        cube.origin[1] + pos[1] * cube.size[1] + (pos[1] ? inflate : -inflate),
        cube.origin[2] + pos[2] * cube.size[2] + (pos[2] ? inflate : -inflate)
      )

      vecPos = vecPos.applyEuler(cubeRotation)
      vecPos = vecPos.sub(bone.position)
      vecPos = vecPos.applyEuler(bone.rotation)
      vecPos = vecPos.add(bone.position)

      attr.positions.push(vecPos.x, vecPos.y, vecPos.z)
      attr.normals.push(...dir)
      attr.uvs.push(u, v)
      attr.skinIndices.push(boneId, 0, 0, 0)
      attr.skinWeights.push(1, 0, 0, 0)
    }

    attr.indices.push(
      ndx, ndx + 1, ndx + 2,
      ndx + 2, ndx + 1, ndx + 3
    )
  }
}

function getMesh (texture, jsonModel) {
  const bones = {}

  const geoData = {
    positions: [],
    normals: [],
    uvs: [],
    indices: [],
    skinIndices: [],
    skinWeights: []
  }
  let i = 0
  for (const jsonBone of jsonModel.bones) {
    const bone = new THREE.Bone()
    if (jsonBone.pivot) {
      bone.position.x = jsonBone.pivot[0]
      bone.position.y = jsonBone.pivot[1]
      bone.position.z = jsonBone.pivot[2]
    }
    if (jsonBone.bind_pose_rotation) {
      bone.rotation.x = -jsonBone.bind_pose_rotation[0] * Math.PI / 180
      bone.rotation.y = -jsonBone.bind_pose_rotation[1] * Math.PI / 180
      bone.rotation.z = -jsonBone.bind_pose_rotation[2] * Math.PI / 180
    } else if (jsonBone.rotation) {
      bone.rotation.x = -jsonBone.rotation[0] * Math.PI / 180
      bone.rotation.y = -jsonBone.rotation[1] * Math.PI / 180
      bone.rotation.z = -jsonBone.rotation[2] * Math.PI / 180
    }
    bones[jsonBone.name] = bone

    if (jsonBone.cubes) {
      for (const cube of jsonBone.cubes) {
        addCube(geoData, i, bone, cube, jsonModel.texturewidth, jsonModel.textureheight)
      }
    }
    i++
  }

  const rootBones = []
  for (const jsonBone of jsonModel.bones) {
    if (jsonBone.parent) bones[jsonBone.parent].add(bones[jsonBone.name])
    else rootBones.push(bones[jsonBone.name])
  }

  const skeleton = new THREE.Skeleton(Object.values(bones))

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(geoData.positions, 3))
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(geoData.normals, 3))
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(geoData.uvs, 2))
  geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(geoData.skinIndices, 4))
  geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(geoData.skinWeights, 4))
  geometry.setIndex(geoData.indices)

  const material = new THREE.MeshLambertMaterial({ transparent: true, skinning: true, alphaTest: 0.1 })
  const mesh = new THREE.SkinnedMesh(geometry, material)
  mesh.add(...rootBones)
  mesh.bind(skeleton)
  mesh.scale.set(1 / 16, 1 / 16, 1 / 16)

  loadTexture(texture, texture => {
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.flipY = false
    texture.wrapS = THREE.RepeatWrapping
    texture.wrapT = THREE.RepeatWrapping
    material.map = texture
  })

  return mesh
}

function get_texture_location(type){
  let texture = ""
  //const texture = "textures/items/" + type
  let texture_minecraft_location = item_names[type].texture
  if (!texture_minecraft_location){
     throw new Error("air was dropped, can not show air as entity")
  }
  let is_item = false
  texture_minecraft_location = texture_minecraft_location.replace("minecraft:", "")

  const block_type = texture_minecraft_location.substring(0,5)

  if (block_type === "block"){
    texture = "textures/blocks/" + texture_minecraft_location.substring(6)
  }
  else {
    texture = "textures/items/" + texture_minecraft_location.substring(6)
  }
  return texture
}

class Entity {
  constructor (version, type, item_count, scene, objectData) {
    const e = entities[type]

    //todo: glow squid in entities.json is just a copy of squid
    if (!e){
      try{
        if (type === "falling_block"){
          const texture = "textures/blocks/" + blocks_with_ids[objectData].name.toLowerCase()
          this.mesh = new THREE.Object3D()

          const mesh = getMesh(texture.replace('textures', 'textures/' + version) + '.png', full_block)
          this.mesh.add(mesh)
        }
        else {
          const texture = get_texture_location(type)
          let tmpJson = dropped_block
          let visual_block_stack = []
          for (let i = 0; i < item_count; i++){
            visual_block_stack.push(block_stack_list[i])
            }
          tmpJson.bones[0].cubes = visual_block_stack
          this.mesh = new THREE.Object3D()

          const mesh = getMesh(texture.replace('textures', 'textures/' + version) + '.png', tmpJson)
          this.mesh.add(mesh)
          }
      }
      catch (e) {
          console.error("missing Entity: " + type)
        console.log(objectData)
      }

      return
    }

    this.mesh = new THREE.Object3D()
    for (const [name, jsonModel] of Object.entries(e.geometry)) {
      let texture = e.textures[name]
      if (!texture){
        const first_key = Object.keys(e.textures)[0]
        texture = e.textures[first_key]

      }
      // console.log(JSON.stringify(jsonModel, null, 2))
      const mesh = getMesh(texture.replace('textures', 'textures/' + version) + '.png', jsonModel)
      /* const skeletonHelper = new THREE.SkeletonHelper( mesh )
      skeletonHelper.material.linewidth = 2
      scene.add( skeletonHelper ) */
      this.mesh.add(mesh)
    }
  }
}

module.exports = Entity
