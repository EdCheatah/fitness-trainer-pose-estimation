document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const exerciseGrid = document.getElementById('exercise-grid');
    const exerciseCount = document.getElementById('exercise-count');
    const exerciseInfo = document.getElementById('exercise-info');
    const infoName = document.getElementById('info-name');
    const infoType = document.getElementById('info-type');
    const infoDescription = document.getElementById('info-description');
    const infoNote = document.getElementById('info-note');
    const categoryTabs = document.querySelectorAll('.category-tab');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const setsInput = document.getElementById('sets');
    const repsInput = document.getElementById('reps');
    const currentExercise = document.getElementById('current-exercise');
    const currentSet = document.getElementById('current-set');
    const currentReps = document.getElementById('current-reps');
    const formScore = document.getElementById('form-score');
    const formGrade = document.getElementById('form-grade');
    
    // Camera elements
    const videoElement = document.getElementById('video');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const videoContainer = document.querySelector('.video-container');

    // Camera state
    let cameraStarted = false;
    
    // Variables
    let selectedExercise = null;
    let exercisesData = {};
    let workoutRunning = false;
    let statusCheckInterval = null;
    let currentCategory = 'all';
    
    // ==================== Camera Control Functions ====================
    function startCamera() {
        if (cameraStarted) return;

        console.log('Starting camera...');
        videoElement.src = '/video_feed';
        videoElement.style.display = 'block';
        if (videoPlaceholder) {
            videoPlaceholder.style.display = 'none';
        }
        if (fullscreenBtn) fullscreenBtn.style.display = 'block';
        cameraStarted = true;
    }

    function stopCamera() {
        if (!cameraStarted) return;

        console.log('Stopping camera...');
        videoElement.src = '';
        videoElement.style.display = 'none';
        if (videoPlaceholder) {
            videoPlaceholder.style.display = 'flex';
        }
        if (fullscreenBtn) fullscreenBtn.style.display = 'none';
        cameraStarted = false;

        // Notify server to release camera
        fetch('/stop_camera', { method: 'POST' }).catch(() => {});
    }

    // Fullscreen toggle
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', function() {
            if (!document.fullscreenElement) {
                videoContainer.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen();
            }
        });

        document.addEventListener('fullscreenchange', function() {
            fullscreenBtn.textContent = document.fullscreenElement ? '✕' : '⛶';
        });
    }
    
    // Start camera button click
    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', startCamera);
    }
    
    // Stop camera when leaving page
    window.addEventListener('beforeunload', stopCamera);
    
    // Also stop camera when clicking navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            stopCamera();
        });
    });
    
    // Exercises hidden from the UI (YAMLs kept, just not shown)
    const HIDDEN_EXERCISES = [
        'mountain_climber', 'plank', 'high_knees', 'deadlift',
        'calf_raise', 'lunge', 'side_lunge', 'wall_sit',
        'shoulder_press', 'hammer_curl', 'tricep_dip', 'push_up'
    ];

    // Exercise categories mapping
    const exerciseCategories = {
        upper: ['bicep_curl', 'lateral_raise'],
        lower: ['squat', 'glute_bridge', 'leg_raise'],
        cardio: ['jumping_jack']
    };
    
    // Exercise type info
    const exerciseTypeInfo = {
        'bilateral': {
            label: 'Bilateral',
            color: '#9b59b6',
            note: 'This exercise alternates between left and right sides. Each side counts separately (e.g., left curl = 1, right curl = 2).'
        },
        'duration': {
            label: 'Duration',
            color: '#e67e22',
            note: 'This is a timed hold exercise. The counter shows seconds held in correct position.'
        },
        'standard': {
            label: 'Standard',
            color: '#3498db',
            note: 'Standard repetition exercise. Each complete movement cycle counts as 1 rep.'
        }
    };
    
    // SVG icons for exercise cards
    const ICON_UPPER = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="9" width="4" height="6" rx="1"/><rect x="18" y="9" width="4" height="6" rx="1"/><rect x="6" y="7" width="2" height="10" rx="1"/><rect x="16" y="7" width="2" height="10" rx="1"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`;
    const ICON_LOWER = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`;
    const ICON_CARDIO = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`;

    // Exercise display names and descriptions
    const exerciseDetails = {
        'squat':           { name: 'Squat',          desc: 'Lower body compound movement targeting quads and glutes', icon: ICON_LOWER },
        'push_up':         { name: 'Push Up',         desc: 'Upper body pushing exercise for chest and triceps',      icon: ICON_UPPER },
        'hammer_curl':     { name: 'Hammer Curl',     desc: 'Bicep curl with neutral grip, alternating arms',         icon: ICON_UPPER },
        'bicep_curl':      { name: 'Bicep Curl',      desc: 'Classic bicep exercise alternating between arms',        icon: ICON_UPPER },
        'tricep_dip':      { name: 'Tricep Dip',      desc: 'Chair/bench dips for tricep strength',                  icon: ICON_UPPER },
        'shoulder_press':  { name: 'Shoulder Press',  desc: 'Overhead pressing for shoulder development',            icon: ICON_UPPER },
        'lateral_raise':   { name: 'Lateral Raise',   desc: 'Side raises for shoulder width',                        icon: ICON_UPPER },
        'lunge':           { name: 'Lunge',           desc: 'Forward lunge alternating between legs',                icon: ICON_LOWER },
        'side_lunge':      { name: 'Side Lunge',      desc: 'Lateral lunge for inner thigh and glutes',              icon: ICON_LOWER },
        'deadlift':        { name: 'Deadlift',        desc: 'Hip hinge movement for hamstrings and back',            icon: ICON_LOWER },
        'glute_bridge':    { name: 'Glute Bridge',    desc: 'Hip bridge for glute activation',                       icon: ICON_LOWER },
        'calf_raise':      { name: 'Calf Raise',      desc: 'Standing raises for calf muscles',                      icon: ICON_LOWER },
        'wall_sit':        { name: 'Wall Sit',        desc: 'Isometric hold against wall for quad endurance',        icon: ICON_LOWER },
        'leg_raise':       { name: 'Leg Raise',       desc: 'Lying leg raises for lower abs',                        icon: ICON_LOWER },
        'plank':           { name: 'Plank',           desc: 'Core stabilization hold exercise',                      icon: ICON_LOWER },
        'mountain_climber':{ name: 'Mountain Climber',desc: 'Dynamic cardio with alternating knee drives',           icon: ICON_CARDIO },
        'high_knees':      { name: 'High Knees',      desc: 'Running in place with high knee lifts',                 icon: ICON_CARDIO },
        'jumping_jack':    { name: 'Jumping Jack',    desc: 'Classic full body cardio movement',                     icon: ICON_CARDIO }
    };
    
    // Load exercises from API
    async function loadExercises() {
        try {
            const response = await fetch('/exercises');
            const data = await response.json();
            exercisesData = data;
            exerciseCount.textContent = `(${data.count} exercises)`;
            renderExercises(data.exercises);
        } catch (error) {
            console.error('Error loading exercises:', error);
            exerciseGrid.innerHTML = '<div class="error">Failed to load exercises</div>';
        }
    }
    
    // Render exercise grid
    function renderExercises(exercises) {
        exerciseGrid.innerHTML = '';
        
        let filteredExercises = exercises.filter(ex => !HIDDEN_EXERCISES.includes(ex));
        if (currentCategory !== 'all') {
            filteredExercises = filteredExercises.filter(ex => exerciseCategories[currentCategory]?.includes(ex));
        }
        
        filteredExercises.forEach(exercise => {
            const info = exercisesData.info[exercise] || {};
            const details = exerciseDetails[exercise] || { name: exercise.replace(/_/g, ' '), desc: '', icon: '🏋️' };
            const typeInfo = exerciseTypeInfo[info.type] || exerciseTypeInfo['standard'];
            
            const div = document.createElement('div');
            div.className = 'exercise-option';
            div.setAttribute('data-exercise', exercise);
            div.setAttribute('data-type', info.type || 'standard');
            
            div.innerHTML = `
                <div class="exercise-icon">${details.icon || ''}</div>
                <h3>${details.name}</h3>
                <span class="exercise-type-badge" style="background-color: ${typeInfo.color}">${typeInfo.label}</span>
            `;
            
            div.addEventListener('click', () => selectExercise(exercise, info));
            exerciseGrid.appendChild(div);
        });
    }
    
    // Select exercise
    function selectExercise(exercise, info) {
        // Remove previous selection
        document.querySelectorAll('.exercise-option').forEach(opt => opt.classList.remove('selected'));
        
        // Add selection to clicked
        const selected = document.querySelector(`[data-exercise="${exercise}"]`);
        if (selected) selected.classList.add('selected');
        
        selectedExercise = exercise;
        
        // Show exercise info
        const details = exerciseDetails[exercise] || { name: exercise, desc: '' };
        const typeInfo = exerciseTypeInfo[info.type] || exerciseTypeInfo['standard'];
        
        infoName.textContent = details.name;
        infoType.textContent = typeInfo.label;
        infoType.style.backgroundColor = typeInfo.color;
        infoDescription.textContent = details.desc;
        infoNote.innerHTML = typeInfo.note;
        
        exerciseInfo.classList.remove('hidden');
    }
    
    // Category tab switching
    categoryTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            categoryTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            currentCategory = this.getAttribute('data-category');
            renderExercises(exercisesData.exercises || []);
        });
    });
    
    // Start workout
    startBtn.addEventListener('click', function() {
        if (!selectedExercise) {
            alert('Please select an exercise first!');
            return;
        }
        
        const sets = parseInt(setsInput.value);
        const reps = parseInt(repsInput.value);
        
        if (isNaN(sets) || sets < 1 || isNaN(reps) || reps < 1) {
            alert('Please enter valid numbers for sets and repetitions.');
            return;
        }
        
        // Start camera first if not started
        startCamera();
        
        // Start the exercise via API
        fetch('/start_exercise', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                exercise_type: selectedExercise,
                sets: sets,
                reps: reps
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                workoutRunning = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
                // Update UI
                const details = exerciseDetails[selectedExercise] || { name: selectedExercise };
                currentExercise.textContent = details.name;
                currentSet.textContent = `1 / ${sets}`;
                currentReps.textContent = `0 / ${reps}`;
                formScore.textContent = '100';
                formGrade.textContent = 'A';
                formGrade.className = 'status-value form-grade grade-a';
                
                // Start status polling
                statusCheckInterval = setInterval(checkStatus, 500);
            } else {
                alert('Failed to start exercise: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while starting the exercise.');
        });
    });
    
    // Stop workout
    stopBtn.addEventListener('click', function() {
        fetch('/stop_exercise', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                resetWorkoutUI();
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
    
    // Function to check status
    function checkStatus() {
        fetch('/get_status')
        .then(response => response.json())
        .then(data => {
            if (!data.exercise_running && workoutRunning) {
                // Workout has ended
                resetWorkoutUI();
                return;
            }
            
            // Update status display
            currentSet.textContent = `${data.current_set} / ${data.total_sets}`;
            currentReps.textContent = `${data.current_reps} / ${data.rep_goal}`;
            
            // Update form score
            if (data.form_score !== undefined) {
                formScore.textContent = Math.round(data.avg_form_score || data.form_score);
                const grade = data.form_grade || getGrade(data.form_score);
                formGrade.textContent = grade;
                formGrade.className = `status-value form-grade grade-${grade.toLowerCase()}`;
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
        });
    }
    
    // Get grade from score
    function getGrade(score) {
        if (score >= 90) return 'A';
        if (score >= 80) return 'B';
        if (score >= 70) return 'C';
        if (score >= 60) return 'D';
        return 'F';
    }
    
    // Reset UI after workout ends
    function resetWorkoutUI() {
        workoutRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
        
        currentExercise.textContent = 'None';
        currentSet.textContent = '0 / 0';
        currentReps.textContent = '0 / 0';
        formScore.textContent = '--';
        formGrade.textContent = '--';
        formGrade.className = 'status-value form-grade';
    }
    
    // Initialize
    loadExercises();
});
