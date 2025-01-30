
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_CATEGORIES = {'cats', 'dogs'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_category(category):
    """Check if category selected."""
    return category in ALLOWED_CATEGORIES