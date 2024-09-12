// ! TEMP {
let imageSequences = [
  {
    itemid: 7144878,
    figure1: 11,
    figure2: 35,
    figure3: 27,
    figure4: 27,
    figure5: 35,
    figure6: 11,
    figure7: 11,
    solution: 35,
    choice1: 11,
    choice2: 35,
    choice3: 19,
    choice4: 27,
    seq_order: "6234150",
    choice_order: "2103",
    pattern: "ABCCBAAB",
  },
  {
    itemid: 10279778,
    figure1: 15,
    figure2: 33,
    figure3: 47,
    figure4: 47,
    figure5: 15,
    figure6: 33,
    figure7: 47,
    solution: 47,
    choice1: 33,
    choice2: 47,
    choice3: 15,
    choice4: 50,
    seq_order: "6054132",
    choice_order: "0213",
    pattern: "ABCCABCC",
  },
  {
    itemid: 4482810,
    figure1: 11,
    figure2: 35,
    figure3: 33,
    figure4: 16,
    figure5: 23,
    figure6: 55,
    figure7: 11,
    solution: 35,
    choice1: 35,
    choice2: 16,
    choice3: 11,
    choice4: 33,
    seq_order: "4165302",
    choice_order: "0321",
    pattern: "ABCDEFAB",
  },
  {
    itemid: 2613552,
    figure1: 16,
    figure2: 35,
    figure3: 16,
    figure4: 35,
    figure5: 33,
    figure6: 52,
    figure7: 33,
    solution: 52,
    choice1: 33,
    choice2: 52,
    choice3: 16,
    choice4: 35,
    seq_order: "4523016",
    choice_order: "3210",
    pattern: "ABABCDCD",
  },
  {
    itemid: 7500779,
    figure1: 33,
    figure2: 16,
    figure3: 27,
    figure4: 33,
    figure5: 33,
    figure6: 16,
    figure7: 27,
    solution: 33,
    choice1: 33,
    choice2: 27,
    choice3: 52,
    choice4: 16,
    seq_order: "4213650",
    choice_order: "3120",
    pattern: "ABCAABCA",
  },
  {
    itemid: 11275993,
    figure1: 53,
    figure2: 53,
    figure3: 53,
    figure4: 53,
    figure5: 53,
    figure6: 53,
    figure7: 53,
    solution: 53,
    choice1: 30,
    choice2: 15,
    choice3: 53,
    choice4: 27,
    seq_order: "1045623",
    choice_order: "0312",
    pattern: "AAAAAAAA",
  },
  {
    itemid: 5169422,
    figure1: 34,
    figure2: 53,
    figure3: 34,
    figure4: 53,
    figure5: 35,
    figure6: 27,
    figure7: 35,
    solution: 27,
    choice1: 53,
    choice2: 34,
    choice3: 27,
    choice4: 35,
    seq_order: "4126035",
    choice_order: "2031",
    pattern: "ABABCDCD",
  },
  {
    itemid: 17095278,
    figure1: 34,
    figure2: 15,
    figure3: 11,
    figure4: 27,
    figure5: 52,
    figure6: 52,
    figure7: 27,
    solution: 11,
    choice1: 11,
    choice2: 34,
    choice3: 15,
    choice4: 52,
    seq_order: "2140635",
    choice_order: "2301",
    pattern: "ABCDEEDC",
  },
  {
    itemid: 4467758,
    figure1: 52,
    figure2: 52,
    figure3: 16,
    figure4: 55,
    figure5: 52,
    figure6: 52,
    figure7: 16,
    solution: 55,
    choice1: 37,
    choice2: 55,
    choice3: 52,
    choice4: 16,
    seq_order: "3461502",
    choice_order: "3021",
    pattern: "AABCAABC",
  },
  {
    itemid: 16717726,
    figure1: 34,
    figure2: 47,
    figure3: 53,
    figure4: 53,
    figure5: 47,
    figure6: 34,
    figure7: 34,
    solution: 47,
    choice1: 14,
    choice2: 47,
    choice3: 34,
    choice4: 53,
    seq_order: "1026354",
    choice_order: "2031",
    pattern: "ABCCBAAB",
  },
];

let imageMappings = {
  0: "alarm-clock",
  1: "apps",
  2: "baby-carriage",
  3: "bell",
  4: "biking",
  5: "bone",
  6: "box-open",
  7: "brightness",
  8: "broadcast-tower",
  9: "bulb",
  10: "camera",
  11: "candy-cane",
  12: "carrot",
  13: "chess",
  14: "club",
  15: "cocktail-alt",
  16: "cube",
  17: "diamond",
  18: "eye",
  19: "fish",
  20: "gamepad",
  21: "gift",
  22: "globe",
  23: "graduation-cap",
  24: "guitar",
  25: "hammer",
  26: "hand-horns",
  27: "headphones",
  28: "heart",
  29: "helicopter-side",
  30: "home",
  31: "ice-skate",
  32: "island-tropical",
  33: "key",
  34: "lock",
  35: "megaphone",
  36: "mug-hot-alt",
  37: "music-alt",
  38: "paper-plane",
  39: "paw",
  40: "peach",
  41: "phone-call",
  42: "plane-alt",
  43: "playing-cards",
  44: "pyramid",
  45: "rocket",
  46: "rugby",
  47: "search",
  48: "settings",
  49: "shopping-basket",
  50: "shopping_cart",
  51: "skiing",
  52: "smile",
  53: "social-network",
  54: "spade",
  55: "star",
  56: "trophy-star",
  57: "truck-side",
  58: "user",
  59: "wheat",
};

// console.log(practiceSequences); // ! TEMP: Debugging
// ! } TEMP

//  * ############### CONFIGURATION ###############
let config = {
  debugLvl: 3,
  serverEndpoint:
    "https://3af36fcb-1736-4dde-8674-e8c00154e141.mock.pstmn.io/data",
  interStimTime: 800,
  feedbackTime: 2000,
  maxRespTime: 500000, // 500000,
  postPracticeDelay: 3000,
  validKeys: ["a", "x", "m", "l"],
};

let imagePaths = [
  "images/resized/shopping_cart.png",
  "images/resized/ice-skate.png",
  "images/resized/headphones.png",
  "images/resized/cube.png",
  "images/resized/peach.png",
  "images/resized/bell.png",
  "images/resized/alarm-clock.png",
  "images/resized/apps.png",
  "images/resized/rugby.png",
  "images/resized/settings.png",
  "images/resized/key.png",
  "images/resized/island-tropical.png",
  "images/resized/lock.png",
  "images/resized/wheat.png",
  "images/resized/playing-cards.png",
  "images/resized/spade.png",
  "images/resized/paw.png",
  "images/resized/chess.png",
  "images/resized/box-open.png",
  "images/resized/carrot.png",
  "images/resized/skiing.png",
  "images/resized/cocktail-alt.png",
  "images/resized/home.png",
  "images/resized/user.png",
  "images/resized/mug-hot-alt.png",
  "images/resized/plane-alt.png",
  "images/resized/hand-horns.png",
  "images/resized/megaphone.png",
  "images/resized/bulb.png",
  "images/resized/broadcast-tower.png",
  "images/resized/music-alt.png",
  "images/resized/search.png",
  "images/resized/heart.png",
  "images/resized/gamepad.png",
  "images/resized/social-network.png",
  "images/resized/shopping-basket.png",
  "images/resized/brightness.png",
  "images/resized/rocket.png",
  "images/resized/globe.png",
  "images/resized/gift.png",
  "images/resized/eye.png",
  "images/resized/truck-side.png",
  "images/resized/question-mark.png",
  "images/resized/diamond.png",
  "images/resized/hammer.png",
  "images/resized/bone.png",
  "images/resized/star.png",
  "images/resized/guitar.png",
  "images/resized/graduation-cap.png",
  "images/resized/pyramid.png",
  "images/resized/phone-call.png",
  "images/resized/club.png",
  "images/resized/camera.png",
  "images/resized/baby-carriage.png",
  "images/resized/paper-plane.png",
  "images/resized/trophy-star.png",
  "images/resized/candy-cane.png",
  "images/resized/smile.png",
  "images/resized/fish.png",
  "images/resized/helicopter-side.png",
  "images/resized/biking.png",
];

let images = preloadImages(imagePaths);
log(4, images);

let midRowText = document.getElementById("mid-row-text");
let topRowId = "top-row";
let midRowId = "middle-row";
let bottomRowId = "bottom-row";

let responses = {
  practice: [],
  main: [],
};

// * ############### FUNCTIONS ###############
function log(lvl, ...message) {
  if (lvl <= config.debugLvl) {
    console.log(...message);
  }
}

function processSequences(sequences, imageMappings) {
  let processedSequences = [];
  for (let sequence of sequences) {
    // Dynamically get top images
    let topImages = [];
    for (let i = 1; i <= 7; i++) {
      topImages.push(sequence[`figure${i}`]);
    }
    topImages = topImages.map((i) => imageMappings[i]); // Map through imageMappings

    // Dynamically get bottom images
    let bottomImages = [];
    for (let i = 1; i <= 4; i++) {
      bottomImages.push(sequence[`choice${i}`]);
    }
    bottomImages = bottomImages.map((i) => imageMappings[i]); // Map through imageMappings

    let sequenceData = {
      ID: sequence.itemid,
      topImages: topImages,
      bottomImages: bottomImages,
      sequenceOrder: sequence.seq_order.split("").map((x) => parseInt(x)),
      choiceOrder: sequence.choice_order.split("").map((x) => parseInt(x)),
      solution: imageMappings[sequence.solution],
    };
    processedSequences.push(sequenceData);
  }
  return processedSequences;
}

function randInt(min, max) {
  // * min and max included
  return Math.floor(Math.random() * (max - min + 1) + min);
}

function formatString(template, ...values) {
  return template.replace(/{}/g, () => values.shift());
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function preloadImages(imagePaths) {
  const images = {};
  for (let path of imagePaths) {
    const img = new Image();
    img.src = path;
    // * Extract the image name without the extension as the key
    // ! Will break if image name has multiple dots TODO: fix this
    const imageName = path.split("/").pop().split(".")[0];
    // * Use the imageName as the key and the Image object as the value
    images[imageName] = img;
  }
  return images;
}

function initializeImageRows() {
  let baseImage = "images/blank_image.png";
  let questionMarkImg = "images/resized/question-mark.png";

  const topRowImages = new Array(7).fill(baseImage).concat([questionMarkImg]);
  const bottomRowImages = new Array(4).fill(baseImage);

  // * Function to add images to a row
  const addImagesToRow = (rowId, imageSources) => {
    const row = document.getElementById(rowId);
    imageSources.forEach((src) => {
      const img = document.createElement("img");
      img.src = src;
      img.alt = ""; // * Set an appropriate alt text for each image
      row.appendChild(img);
    });
  };

  addImagesToRow("top-row", topRowImages);
  addImagesToRow("bottom-row", bottomRowImages);
}

function setAllImages(state = "show") {
  const topRow = document.querySelectorAll("#" + topRowId + " img");
  const bottomRow = document.querySelectorAll("#" + bottomRowId + " img");
  const allImages = [...topRow, ...bottomRow];
  if (state === "show") {
    allImages.forEach((img) => (img.style.opacity = "1"));
  } else if (state === "hide") {
    allImages.forEach((img) => (img.style.opacity = "0"));
  } else {
    console.error(
      "Invalid state argument for setAllImages(). Use 'show' or 'hide'"
    );
  }
}

async function displayImagesSequentially(
  rowID,
  imageSources,
  waitTime,
  indices
) {
  return new Promise(async (resolve) => {
    // log(1, "row:", rowID, "order:", indices);

    let currentImages = document.querySelectorAll("#" + rowID + " img");

    imageSources.forEach((img, idx) => {
      currentImages[idx].src = img;
    });

    for (const idx of indices) {
      currentImages[idx].style.opacity = "1";
      await delay(waitTime);
      currentImages[idx].style.opacity = "0";
    }
    resolve();
  });
}

async function startImageSequence(
  imageSequences,
  trial_type = "main",
  interTrialInterval = [1000, 3000]
) {
  trial_type = ["practice", "main"].includes(trial_type) ? trial_type : "main";

  for (let i = 0; i < imageSequences.length; i++) {
    // * Generate a random wait period between 1 and 3 seconds
    const interTrialTime = randInt(
      interTrialInterval[0],
      interTrialInterval[1]
    );

    log(3, "WaitTime:", interTrialTime);

    // * Wait for the random period before starting the sequence
    await new Promise((resolve) => setTimeout(resolve, interTrialTime));

    // console.log(imageSequences[i]) // ! TEMP: Debugging

    let topRowImages = imageSequences[i].topImages.map(
      (imageName) => images[imageName].src
    );
    let bottomRowImages = imageSequences[i].bottomImages.map(
      (imageName) => images[imageName].src
    );
    // let solution = topRowImages.pop().split("/").pop();
    let solution = imageSequences[i].solution;
    let sequenceOrder = imageSequences[i].sequenceOrder;
    sequenceOrder.push(7);

    let choiceOrder = imageSequences[i].choiceOrder;

    await displayImagesSequentially(
      topRowId,
      topRowImages,
      config.interStimTime,
      sequenceOrder
    );
    await displayImagesSequentially(
      bottomRowId,
      bottomRowImages,
      config.interStimTime,
      choiceOrder
    );

    midRowText.textContent = "";
    setAllImages("show");
    const startTime = Date.now(); // * time right before waiting for a keypress

    const {
      key: KeyPressed,
      index: KeyIndex,
      respTime,
    } = await getResponse(
      startTime,
      config.validKeys,
      config.feedbackTime,
      config.maxRespTime
    );

    let selectedImage;
    let correct;
    // * Check if KeyIndex is not -1 to determine if the key is valid
    if (KeyIndex !== -1) {
      selectedImage = bottomRowImages[KeyIndex].split("/")
        .pop()
        .replace(".png", "");
      correct = solution === selectedImage;
    } else {
      selectedImage = "invalid";
      correct = "invalid";
    }

    if (trial_type === "practice") {
      const bottomImagesElements = document.querySelectorAll(
        "#" + bottomRowId + " img"
      );

      const correctIndex = bottomRowImages.findIndex(
        (img) => img.split("/").pop().replace(".png", "") === solution
      );
      // console.log("Solution:", solution) // ! TEMP: Debugging
      // console.log(bottomRowImages) // ! TEMP: Debugging
      // console.log("correctIndex:", correctIndex) // ! TEMP: Debugging
      bottomImagesElements[correctIndex].classList.add("correct-selection");

      // * Apply feedback based on the correctness
      if (correct === false) {
        bottomImagesElements[KeyIndex].classList.add("incorrect-selection");
      }

      await delay(config.feedbackTime);

      bottomImagesElements.forEach((el) => {
        el.classList.remove("correct-selection", "incorrect-selection");
      });
    }

    let trial_data = {
      sequenceN: i,
      sequenceID: imageSequences[i].ID,
      keyPressed: KeyPressed,
      keyIndex: KeyIndex,
      selectedImage: selectedImage,
      respTime: respTime,
      correct: correct,
      solution: solution,
    };

    log(3, "Trial Data:", trial_data); // ! TEMP: testing

    responses[trial_type].push(trial_data);

    sendDataToServer(trial_data); // ! TEMP: testing

    setAllImages("hide");
    midRowText.textContent = "+";
  }
}

function getResponse(
  startTime,
  validKeys = ["a", "x", "m", "l"],
  feedbackTime = 2000,
  maxRespTime = null
) {
  return new Promise((resolve) => {
    let timeoutHandler; // * To store the timeout that waits for config.maxRespTime

    const keyHandler = (event) => {
      clearTimeout(timeoutHandler); // * Clear the timeout because a key was pressed
      const key = event.key.toLowerCase();
      const endTime = Date.now(); // * Capture the end time at the moment of key press
      const respTime = endTime - startTime; // * Calculate the response time

      if (config.validKeys.includes(key)) {
        document.removeEventListener("keydown", keyHandler); // * Remove the event listener for valid keys
        const index = config.validKeys.indexOf(key);
        const resp_data = { key: key, index: index, respTime: respTime };
        resolve(resp_data);
      } else {
        // * Handle invalid key press
        const resp_data = {
          key: formatString("invalid:[{}]", key),
          index: -1,
          respTime: respTime,
        };

        setAllImages("hide");

        var midRowText = document.getElementById("mid-row-text");

        midRowText.textContent =
          "Invalid Key Pressed! Please press a valid key (a, x, m, l)";
        // await delay(config.feedbackTime);
        // midRowText.textContent = "+"; // * Clear the message
        // document.removeEventListener("keydown", keyHandler); // * Remove the event listener after timeout
        // resolve(resp_data);
        setTimeout(() => {
          midRowText.textContent = "+"; // * Clear the message
          document.removeEventListener("keydown", keyHandler); // * Remove the event listener after timeout
          resolve(resp_data); // * Also include respTime for invalid keys
        }, config.feedbackTime);
      }
    };

    document.addEventListener("keydown", keyHandler);

    // * Set a timeout to enforce max response time
    timeoutHandler = setTimeout(() => {
      document.removeEventListener("keydown", keyHandler); // * Remove the event listener since time is up
      const resp_data = { key: "invalid", index: -1, respTime: -1 };
      resolve(resp_data); // * Resolve with a timeout indication
    }, config.maxRespTime);
  });
}

async function sendDataToServer(data) {
  try {
    const response = await fetch(config.serverEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    const jsonResponse = await response.json();
    log(2, "Server response:", jsonResponse);
  } catch (error) {
    console.error("Error sending data to server:", error);
  }
}


function main() {
  let mainSequences = processSequences(imageSequences, imageMappings);
  let practiceSequences = mainSequences.splice(0, 2); // ! TEMP: For testing
  
  midRowText.textContent = "Welcome to the experiment! Press the spacebar to continue.";

  document.addEventListener("DOMContentLoaded", (event) => {
    initializeImageRows();
  });
  
  document.addEventListener("keydown", async function (event) {
    if (event.code === "Space") {
      // * PRACTICE TRIALS
      let instr1 =
        "You are going to solve __ abstract reasoning problems like the one below. " + 
        "Your goal is to continue the sequence in the top row with one of the four " +
        "options in the bottom row. \nUse the keys a, x, m, l to select one of these " +
        "options from left to right. \nYou will perform two practice trials with " + 
        "feedback before the start of the experiment. \n" +
        "Place your fingers on the " + formatString("{}, {}, {}, {} ", ...config.validKeys) +
        "keys and press any of them to start the practice trials.";
        
      midRowText.textContent = instr1;

      // await practiceTrial(practiceSequences);

      await new Promise((resolve) => {
        const keyHandler = (event) => {
          const key = event.key.toLowerCase();
          if (config.validKeys.includes(key)) {
            // * Remove the event listener to avoid triggering it multiple times.
            document.removeEventListener("keydown", keyHandler);
            resolve(); // * Resolve the promise to continue execution.
          }
        };
        document.addEventListener("keydown", keyHandler);
      });

      midRowText.textContent = "+"; // * This line might be redundant depending on your intended flow

      const startPracticeTime = Date.now();
      await startImageSequence(practiceSequences, "practice");
      const endPracticeTime = Date.now();

      // * MAIN TRIALS
      let instr2 = 
      "End of the practice trials. \n" +  
      "You are now going to solve __ sequences. You won't receive feedback on your answers anymore.\n" +
      "Be sure to think carefully " +
      "about your response as we plan to compare humans to AI. \n"+
      "Then choose the correct option as quickly as you can.\n" +
      "Place your fingers on the " + formatString("{}, {}, {}, {} ", ...config.validKeys) +
      "keys and press any of them to start the experiment.";
      
      midRowText.textContent = instr2;
      
      await new Promise((resolve) => {
        const keyHandler = (event) => {
          const key = event.key.toLowerCase();
          if (config.validKeys.includes(key)) {
            // * Remove the event listener to avoid triggering it multiple times.
            document.removeEventListener("keydown", keyHandler);
            resolve(); // * Resolve the promise to continue execution.
          }
        };
        document.addEventListener("keydown", keyHandler);
      });
      
      midRowText.textContent = "+"; // * This line might be redundant depending on your intended flow
      
      const startExpTime = Date.now();
      await startImageSequence(mainSequences); // * Correctly await the async function
      const endExpTime = Date.now();
      const expDuration = endExpTime - startExpTime;
    
      let debrief = "End of Experiment. Thank you for participating!";
      midRowText.textContent = debrief
      
      log(1, responses);
      log(0, "Duration:", expDuration / 1000, "s"); 
    }
  });
}

// * ############### LAUNCH EXPERIMENT ###############
main();