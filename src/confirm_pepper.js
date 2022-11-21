const PATH = "./uid_pepper"
const _DEBUG = false

const fs = require('fs')
const { uids, hashes, tags} = require('./uids_hashes.json')
require('./gen_hashcode.js') // expands String.prototype with hashCode() function

// extracts the uid_pepper2 and pepper1 from a string of the
// form '___89087912348901894f'
function extract_values(str) {
  let i = 0
  while (str[i] === "_") i++
  let uid_pepper2 = str.slice(i, -6)

  let pepper1 = str[str.length - 6] === "0" ? str.slice(-5) : str.slice(-6)

  return [pepper1, uid_pepper2]
}

// searches the given uid_pepper2_values for the string that matches the given
// uid_pepper2 and returns the pepper2 that generated this value
function find_pepper2(uid_pepper2, uid_pepper2_values) {
  let i = 0
  // endless loop when uid_pepper2 isn't found
  while (uid_pepper2_values[i] !== uid_pepper2 && i < uid_pepper2_values.length) i++
  if (i >= uid_pepper2_values.length)
    throw "The given string doesn't match any uid_pepper2 in the file: " + uid_pepper2
  // pepper2 starts at 100
  return i + 100
}

function str2pepper(str, uid_pepper2_values) {
  let pepper = extract_values(str)
  pepper[1] = find_pepper2(pepper[1], uid_pepper2_values)

  return pepper
}

function calc_hash(uid, pepper1, pepper2) {
  return (String(Number(uid) * pepper2) + pepper1).hashCode()
}

function compare_hashcode_36(target, calc) {
  const res = target - calc
  return (res < 10 && res >= 0) || (res <= 'z'.charCodeAt(0) - '0'.charCodeAt(0) && res >= 'a'.charCodeAt(0) - '0'.charCodeAt(0));
}

function correct_pepper1(target, calc) {
  return String.fromCharCode('0'.charCodeAt(0) + (target - calc))
}

// read the uid_pepper2_values from the previously generated file
let uid_pepper2_file = fs.readFileSync(PATH).toString()
let uid_pepper2_values = uid_pepper2_file.split('\n')
if (uid_pepper2_values.length !== 900)
  throw 'uid_pepper should contain 900 values, but has ' + uid_pepper2_values.length

function main(str) {
  const pepper = str2pepper(str, uid_pepper2_values)
  if (_DEBUG) console.log('Extracted pepper: ' + pepper)

  let matches = []
  let corr_pepper1
  let err = false
  // uids.forEach((uid, i) => {
  //   let hash = calc_hash(uid, pepper[0], pepper[1])
  //   let j = hashes.findIndex(x => compare_hashcode_36(x, hash))
  //
  //   if (j === -1 && !_DEBUG) // no corresponding hash was found
  //     return err = true
  //   else {
  //     matches.push(tags[j])
  //     corr_pepper1 = pepper[0].slice(0, -1) + correct_pepper1(hashes[j], hash)
  //   }
  // })
  hashes.forEach((target) => {
    let j = uids.findIndex((uid) => compare_hashcode_36(target, calc_hash(uid, pepper[0], pepper[1])))

    if (j === -1 && !_DEBUG) // no corresponding hash was found
      return err = true
    else {
      matches.push(tags[j])
      corr_pepper1 = pepper[0].slice(0, -1) + correct_pepper1(target, calc_hash(uids[j], pepper[0], pepper[1]))
    }
  })
  if (err)
    return false

  if (_DEBUG && matches[matches.length - 1] === undefined) {
    console.log(matches)
    return false
  }

  return [[corr_pepper1, pepper[1]], [tags, matches]]
}


if (require.main === module) {
  // process.argv = ['node', 'confirm_pepper.js', '___89087912348901894f']
  const str = process.argv[2]
  let out = main(str)

  if (!out)
    process.exit(1)

  let pepper = out[0]
  let tags = out[1][0]
  let matches = out[1][1]

  let out_str = tags.map((tag, i) => tag + " -> " + matches[i]).join('\n')
  console.log('Found a result! Pepper: ' + pepper[0] + ", " + pepper[1])
  console.log('\n' + out_str)
}

module.exports = main
