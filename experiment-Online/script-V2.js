// ! TEMP {
let imageSequences = [
  {
    itemid: 20230887,
    figure1: 23,
    figure2: 47,
    figure3: 27,
    figure4: 11,
    figure5: 23,
    figure6: 47,
    figure7: 27,
    figure8: 11,
    maskedImageIdx: 0,
    solution: 23,
    choice1: 11,
    choice2: 23,
    choice3: 27,
    choice4: 47,
    seq_order: "20376451",
    choice_order: "2031",
    pattern: "ABCDABCD",
  },
  {
    itemid: 5238528,
    figure1: 55,
    figure2: 15,
    figure3: 23,
    figure4: 27,
    figure5: 55,
    figure6: 15,
    figure7: 23,
    figure8: 27,
    maskedImageIdx: 7,
    solution: 27,
    choice1: 15,
    choice2: 55,
    choice3: 27,
    choice4: 23,
    seq_order: "57614230",
    choice_order: "1203",
    pattern: "ABCDABCD",
  },
  {
    itemid: 8051903,
    figure1: 16,
    figure2: 34,
    figure3: 34,
    figure4: 16,
    figure5: 16,
    figure6: 34,
    figure7: 34,
    figure8: 16,
    maskedImageIdx: 4,
    solution: 16,
    choice1: 34,
    choice2: 37,
    choice3: 39,
    choice4: 16,
    seq_order: "26107534",
    choice_order: "2310",
    pattern: "ABBAABBA",
  },
  {
    itemid: 2749853,
    figure1: 55,
    figure2: 52,
    figure3: 52,
    figure4: 55,
    figure5: 55,
    figure6: 52,
    figure7: 52,
    figure8: 55,
    maskedImageIdx: 2,
    solution: 52,
    choice1: 52,
    choice2: 41,
    choice3: 12,
    choice4: 55,
    seq_order: "03675214",
    choice_order: "3120",
    pattern: "ABBAABBA",
  },
  {
    itemid: 9652411,
    figure1: 33,
    figure2: 47,
    figure3: 47,
    figure4: 47,
    figure5: 33,
    figure6: 47,
    figure7: 47,
    figure8: 47,
    maskedImageIdx: 3,
    solution: 47,
    choice1: 27,
    choice2: 47,
    choice3: 7,
    choice4: 33,
    seq_order: "26017345",
    choice_order: "0321",
    pattern: "ABBBABBB",
  },
  {
    itemid: 2653830,
    figure1: 35,
    figure2: 34,
    figure3: 11,
    figure4: 35,
    figure5: 35,
    figure6: 34,
    figure7: 11,
    figure8: 35,
    maskedImageIdx: 4,
    solution: 35,
    choice1: 35,
    choice2: 18,
    choice3: 11,
    choice4: 34,
    seq_order: "04326157",
    choice_order: "2031",
    pattern: "ABCAABCA",
  },
  {
    itemid: 3605224,
    figure1: 33,
    figure2: 33,
    figure3: 33,
    figure4: 33,
    figure5: 33,
    figure6: 33,
    figure7: 33,
    figure8: 33,
    maskedImageIdx: 1,
    solution: 33,
    choice1: 46,
    choice2: 20,
    choice3: 36,
    choice4: 33,
    seq_order: "63427501",
    choice_order: "1302",
    pattern: "AAAAAAAA",
  },
  {
    itemid: 7045354,
    figure1: 55,
    figure2: 55,
    figure3: 47,
    figure4: 47,
    figure5: 55,
    figure6: 55,
    figure7: 47,
    figure8: 47,
    maskedImageIdx: 5,
    solution: 55,
    choice1: 5,
    choice2: 55,
    choice3: 40,
    choice4: 47,
    seq_order: "07325146",
    choice_order: "0123",
    pattern: "AABBAABB",
  },
  {
    itemid: 18877209,
    figure1: 11,
    figure2: 23,
    figure3: 23,
    figure4: 23,
    figure5: 11,
    figure6: 23,
    figure7: 23,
    figure8: 23,
    maskedImageIdx: 6,
    solution: 23,
    choice1: 1,
    choice2: 11,
    choice3: 16,
    choice4: 23,
    seq_order: "76420513",
    choice_order: "0123",
    pattern: "ABBBABBB",
  },
  {
    itemid: 60633,
    figure1: 35,
    figure2: 27,
    figure3: 34,
    figure4: 47,
    figure5: 11,
    figure6: 11,
    figure7: 47,
    figure8: 34,
    maskedImageIdx: 4,
    solution: 11,
    choice1: 34,
    choice2: 27,
    choice3: 11,
    choice4: 47,
    seq_order: "53012746",
    choice_order: "0213",
    pattern: "ABCDEEDC",
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

// ! } TEMP

//  * ############### CONFIGURATION ###############
let mainSequences = processSequences(imageSequences, imageMappings);
let practiceSequences = mainSequences.splice(0, 2); // ! TEMP: For testing

let config = {
  debugLvl: 1,
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

let blankImage = "images/blank_image.png";
let questionMarkImg = "images/resized/question-mark.png";
// imagePaths = imagePaths.concat(["images/blank_image.png", "images/resized/question-mark.png"]);
// console.log(imagePaths);

let images = preloadImages(imagePaths);
// let blankImage = images["blank_image"];
// let questionMarkImg = images["question-mark"];

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

    for (let i = 1; i <= 8; i++) {
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
      maskedImageIdx: sequence.maskedImageIdx,
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

  // const topRowImages = new Array(7).fill(blankImage).concat([questionMarkImg]);
  const topRowImages = new Array(8).fill(blankImage);
  const bottomRowImages = new Array(4).fill(blankImage);

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
  seqOrder,
  maskedImageIdx = null
) {
  return new Promise(async (resolve) => {

    let currentImages = document.querySelectorAll("#" + rowID + " img");

    // * If maskedImageIdx not null, replace image at that index with questionMarkImg
    imageSources.forEach((imageName, idx) => {
      // console.log("imageName:", imageName);
      if (idx === maskedImageIdx) {
        currentImages[idx].src = questionMarkImg;
        // currentImages[idx].src = images["question-mark"].src
      } else {
        currentImages[idx].src = imageName;
        // currentImages[idx].src = images[imageName].src;
      }
    });

    // * Move maskedImageIdx to the end of seqOrder => display questionMarkImg last
    if (maskedImageIdx !== null) {
      seqOrder.push(seqOrder.splice(seqOrder.indexOf(maskedImageIdx), 1)[0]);
    };

    // * Display images sequentially
    for (const idx of seqOrder) {
      currentImages[idx].style.opacity = "1";
      await delay(waitTime);
      currentImages[idx].style.opacity = "0";
    }

    // * Display images sequentially if image != maskedImage
    // for (const idx of seqOrder) {
    //   if (idx !== maskedImageIdx) {
    //     currentImages[idx].style.opacity = "1";
    //     await delay(waitTime);
    //     currentImages[idx].style.opacity = "0";
    //   }
    // }

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
    // await new Promise((resolve) => setTimeout(resolve, interTrialTime));
    await delay(interTrialTime);

    let topRowImages = imageSequences[i].topImages.map(
      (imageName) => images[imageName].src
    );

    let bottomRowImages = imageSequences[i].bottomImages.map(
      (imageName) => images[imageName].src
    );
    // let solution = topRowImages.pop().split("/").pop();
    let solution = imageSequences[i].solution;
    let sequenceOrder = imageSequences[i].sequenceOrder;
    let maskedImageIdx = imageSequences[i].maskedImageIdx;
    // sequenceOrder.push(7);

    let choiceOrder = imageSequences[i].choiceOrder;

    await displayImagesSequentially(
      topRowId,
      topRowImages,
      config.interStimTime,
      sequenceOrder,
      maskedImageIdx
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

        let midRowText = document.getElementById("mid-row-text");

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

async function main() {
  midRowText.textContent =
    "Welcome to the experiment! Press the spacebar to continue.";

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
        "Place your fingers on the " +
        formatString("{}, {}, {}, {} ", ...config.validKeys) +
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
        "about your response as we plan to compare humans to AI. \n" +
        "Then choose the correct option as quickly as you can.\n" +
        "Place your fingers on the " +
        formatString("{}, {}, {}, {} ", ...config.validKeys) +
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
      midRowText.textContent = debrief;

      log(1, responses);
      log(0, "Duration:", expDuration / 1000, "s");
    }
  });
}

// * ############### LAUNCH EXPERIMENT ###############
main();
