/**
* Usage: node gen_uid_pepper2.js UID
*/

const PATH = './uid_pepper'

const fs = require('fs')

function main(uid) {
  let out = []
  for (let i = 100; i < 1000; i++)
    out.push(String(uid * i))

  const out_str = out.join('\n')
  fs.writeFileSync(PATH, out_str)
}

if (require.main === module) {
  const uid = Number(process.argv[2])
  // TODO: validify uid
  main(uid)
  console.log('Wrote all uid_pepper2s to file ' + PATH)
}

module.exports = main
