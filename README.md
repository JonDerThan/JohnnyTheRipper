This project is solely for bruteforcing the pairs in [my other project](https://github.com/JonDerThan/wichteln-bot). To make sure every participant was paired to exactly one other user, the bot assigns a hash to each user and [displays that hash publicly](https://github.com/JonDerThan/wichteln-bot/blob/99a6313a4df5feb684191f98b6ff0f14dcc85f07/start_event.js#L117).

The hash is comprised of the partners UID and two peppers that change with each draw but are the same for each of the users in a single draw. Therefore, if two people were assigned the same hash, they also were assigned the same partner.

By bruteforcing these two peppers one can hash the UIDs again with these peppers and therefore conclude the assigned partners/pairs.

----------------------------------

# Technical information
The two peppers will be called `PEPPER1` and `PEPPER2` from here on. [They are created like this:](https://github.com/JonDerThan/wichteln-bot/blob/99a6313a4df5feb684191f98b6ff0f14dcc85f07/start_event.js#L37)
```javascript
const PEPPER1 = (Math.random() + 1).toString(36).substring(7)
const PEPPER2 = Math.floor(Math.random() * 900 + 100)
```
### PEPPER1
The `toString(36)` method converts the previous number to a base36 string. It therefore consists of the symbols `[0-9a-z]`. Through testing I found that the `substring(7)` methods reduces the string to `5-6` characters (in most cases). The "lowest" string is therefore `10000` and the "highest" is `zzzzzz`. There are roughly `36^6` possibilities for the first pepper.

### PEPPER2
The second pepper is a simple integer in the range of `[100-999]`. This results in `900` possible integers.

Overall this results in `36^6 * 900 ≈ 1,96e12` possible combinations. To put this into perspective, a 7 character password with lower-/uppercase letters and numbers (`[0-9a-zA-Z]`) has `62^7 ≈ 3,52e12` possible combinations. Since [the used hash function](https://github.com/JonDerThan/wichteln-bot/blob/99a6313a4df5feb684191f98b6ff0f14dcc85f07/start_event.js#L9) isn't very secure, trying out all possible combinations can be done pretty quickly.

To add to that, the resulting hash only has `32 bits` of information (in JavaScript all bitwise operations are done on 32 bit two's complement integers, which is ensured by [this line](https://github.com/JonDerThan/wichteln-bot/blob/99a6313a4df5feb684191f98b6ff0f14dcc85f07/start_event.js#L15)). [Discord IDs](https://discord.com/developers/docs/reference#snowflakes) have a maximum length of `64 bits`, through the multiplication with `PEPPER2` a maximum of `74 bits` is archived and with the addition of `PEPPER1` (`8 bits * 6`) the maximum amount of information is `122 bits` (in theory).

This leads to a lot of hash collisions, which makes finding a correct solution significantly easier.

---------------------------------------------

## Methodology
1. **Generating uid_pepper:** Since JavaScript has some weird quirks when dealing with large integers, I didn't try to implement any of that in C. Therefore, the first operation of the program consists of generating all 900 combinations of the given UID and `PEPPER2` in JavaScript.
```js
// gen_uid_pepper2.js
// ...
for (let i = 100; i < 1000; i++) out.push(String(uid * i))
// ...
```
The result of this is then saved in a file, `./uid_pepper`.
2. Then, for an already known combination of `(UID, hash)` each combination of `(PEPPER1, PEPPER2)` is tried out. For every tuple the resulting hash is calculated: `hash(UID, PEPPER1, PEPPER2)`.
3. This calculated hash is then compared against the already known hash of the UID. If both hashes match, the combination of the two peppers may be correct *(see note below for further info).* This operation is almost fully parallelizable, so I implemented it with CUDA on the GPU. It starts 10 blocks with 90 threads each, resulting in `900` threads, which is exactly the number of possible combinations for `PEPPER2`. Although the GPU can work with a lot more concurrent threads, I found this was easiest to implement.
4. Since there are a lot of hash collisions, a pepper combination that works with the given `(UID, hash)` tuple may not correctly solve the other UIDs. Because of this, `confirm_pepper.js` takes the calculated pepper combination, calculates `hash(UID, PEPPER1, PEPPER2)` for each UID in the event, and cross-checks them with the actual hashes generated in the event.
5. If the combination of peppers lead to correct hashes for each UID, the last character of `PEPPER1` is corrected *(see note)* and the assigned hashes are resolved for the actual assigned users.

*(Note: Since the hash function [simply adds the last character](https://github.com/JonDerThan/wichteln-bot/blob/99a6313a4df5feb684191f98b6ff0f14dcc85f07/start_event.js#L14) to the resulting hash, which is the last character of `PEPPER1`, the resulting hashes of e.g. `abcd0` and `abcdz` only differ by a small integer. Therefore, only the hashes for {`abcd0`, `abce0`, ...} have to be calculated. This divides the number of actual possible combinations by `36`, while increasing the computing effort for each comparison by a small amount.)*

--------------------------------

# Usage
## Installation
The program itself doesn't have to be installed but it does depend on some other programs.
1. **Microsoft Visual C++ Redistributable:** This is needed by a lot of programs, so it's likely you already have this installed. You can try to run the program without installing it first, if that doesn't work, download the latest release for the `x64` architecture and install it ([link](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022) or [direct download link](https://aka.ms/vs/17/release/vc_redist.x64.exe)).
2. **Node.js:** Download the LTS and install it: [node.js](https://nodejs.org/).
3. **Geforce drivers:** This program depends on Nvidia drivers. I didn't do extensive testing, it presumably needs at least version `526.98` to work correctly. If the program simply stops execution a few seconds after the `Starting calculation...` message, its because of these drivers.

To use the program, simply download the latest release and run `node jtr.js`. The program then asks you for all the details it needs to run, which are explained here in more detail.

1. **UID:** The program needs one user id and the corresponding hash for it to work. You should already know the hash of one user and fill out this field with their user id (e.g. your assigned partner).
2. **Hash:** The hash of the user you used in step 1 (e.g. your assigned hash).
3. **UIDs, Hashes, Tags:** Finally, the program will create a `uids_hashes.json` file for you. You have to open this file and input every user id of the draw, every hash, and every tag (which means the full user identifier, `"User#0001"`). Fill out all fields in the same order as they appear in the group messages of the bot.
4. Confirm with `y` and wait for the program to find a solution.
