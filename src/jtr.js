const UIDS_HASHES_PATH = "./uids_hashes.json"
const EXECUTABLE = './JohnnyTheRipper.exe'

const readline = require('readline')
const fs = require('fs')
const child_process = require('child_process')

const gen_uid_pepper2 = require('./gen_uid_pepper2.js')
let confirm_pepper // initialized elsewhere

const UID_REGEX = /^\d{7,20}$/
function validate_uid(uid) { return UID_REGEX.test(uid) }
function validate_hash(hash) { return hash <= 2147483647 && hash >= -2147483648 }

const UIDS_HASHES_FIELDS = {
  "uids" : [ "uid1", "uid2", "etc" ],
  "hashes": [ 1, 2, 99 ],
  "tags": [ "User1#0001", "User2#1234", "User99#1312" ]
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

function confirm(query, common = false, recursive = false) {
  return new Promise((resolve) => {
    if (!recursive) {
      if (common)
        query += " [Y/n]: "
      else
        query += " [y/N]: "
    }

    rl.question(query, (answer) => {
      switch (answer) {
        case "y":
        case "Y":
          resolve(true)
          break;
        case "n":
        case "N":
          resolve(false)
          break;

        case "":
          resolve(common)
          break

        default:
          confirm(query, common, true).then(resolve)
      }
    })
  })
}

function input(query) {
  return new Promise((resolve) => { rl.question(query, resolve) })
}

const MIN_POS = Number.parseInt("10000", 36)
const MAX_POS = Number.parseInt("ffffff", 36)

function get_pos(str) { return Math.round((Number.parseInt(str.splice(-6), 36) - MIN_POS) / MAX_POS * 10000) / 100 }

/**
* n is the maximum string length, this function generates a number of spaces
* so that |str + spaces| = n
*
* used for aligning multiple strings
*/
function spaces(str, n) {
  let out_str = ""
  for (let i = n - str.length; i > 0; i--) {
    out_str += " "
  }
  return out_str
}

function gen_solution_string(out) {
  let pepper = out[0]
  let tags = out[1][0]
  let matches = out[1][1]

  // n := maximum tag length
  let n = 0
  tags.forEach((tag) => {
    n = Math.max(tag.length, n)
  });

  return  '\n\n---------- Found a result: ' + pepper[0] + ", " + pepper[1] + ' ----------\n\n' +
          tags.map((tag, i) => tag + spaces(tag, n) + " -> " + matches[i]).join('\n')
}

function start(hash) {
  // TODO: display pos and false_positives
  let pos = 0
  let false_positives = 0

  const child_p = child_process.spawn(EXECUTABLE, [hash])

  const rl1 = readline.createInterface({ input: child_p.stdout })
  rl1.on('line', (str) => {
    let out = confirm_pepper(str)

    if (!out)
        false_positives++
    else {
      console.log(gen_solution_string(out))
      child_p.kill()
      // BUG: program prints multiple solutions if the program was quick enough
    }
  })
}

async function main() {
  let uid = await input("Please input the user id: ")
  while (!validate_uid(uid)) {
    uid = await input("Invalid user id, please enter another one: ")
  }
  let hash = Number(await input("Please input the corresponding hash: "))
  while (!validate_hash(hash)) {
    hash = Number(await input("Invalid hash, please enter another one: "))
  }

  console.log("\nNow you need to input all the hashes, ids, and tags in another file.")
  let createFile = true
  if (fs.existsSync(UIDS_HASHES_PATH))
    createFile = await confirm("It seems you already have this file. Do you want to create a new one?", false)

  if (createFile) {
    fs.writeFileSync(UIDS_HASHES_PATH, JSON.stringify(UIDS_HASHES_FIELDS, null, 2))
    console.log("The file was created for you. Please fill out all the fields and confirm.")
    // block until user inputs "y"
    while (!(await confirm("Did you fill out all fields?", false))) { }
  }

  console.log('Generating strings to work on...')
  gen_uid_pepper2(uid)

  // can only be required from here on because it depends on previously generated file
  confirm_pepper = require('./confirm_pepper.js')

  console.log('Starting calculation...')
  start(hash)

  rl.close()
}

if (require.main === module) {
  main()
}
