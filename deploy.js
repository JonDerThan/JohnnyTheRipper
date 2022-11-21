// Deletes temporary files and copys the needed .js files to the correct location

const fs = require('fs')

const BUILD_PATH = './x64/Release/'
const EXECUTABLE = 'JohnnyTheRipper.exe'
// 2d array, first element of each array is dir path, all other are file names
const FILES_TO_COPY = [[
  './src/',
  'confirm_pepper.js',
  'gen_hashcode.js',
  'gen_uid_pepper2.js',
  'jtr.js'
]]

function copy_file(file_names) { fs.copyFileSync(file_names[0], file_names[1]) }

function main() {
  // Deleting temporary files
  const file_names = fs.readdirSync(BUILD_PATH).filter((file_name) => file_name !== EXECUTABLE)
  console.log('Deleting ' + file_names.length + ' temporary files and folders...')
  file_names.forEach((file_name) => {
    fs.rmSync(BUILD_PATH + file_name, { recursive: true })
  })

  // Copying js files
  FILES_TO_COPY.forEach((dir) => {
    // pop the first element which should be the dir_path
    const dir_path = dir.shift()

    // generate 2d array: [[old_file, new_file], [old_file, new_file]]
    let files = dir.map((file_name) => [dir_path + file_name, BUILD_PATH + file_name])
    console.log('Copying ' + files.length + ' files...')
    files.forEach(copy_file)
  })

  console.log('Done.')
}

if (require.main === module)
  main()
