from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from ..models import db, User, Product, Investment, Transaction, SoilReport, GovScheme, SchemeApplication
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_
from functools import wraps
import json
from werkzeug.utils import secure_filename
import os

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    try:
        # Get user statistics with error handling
        total_users = User.query.count()
        total_farmers = User.query.filter_by(role='farmer').count()
        total_investors = User.query.filter_by(role='investor').count()
        
        # Get transaction volume with NULL handling
        total_volume = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0)).scalar()
        
        # Get pending users with validation
        pending_users = User.query.filter(
            User.is_verified == False,
            User.created_at != None
        ).order_by(User.created_at.desc()).all()
        
        # Get recent transactions with validation
        recent_transactions = Transaction.query.filter(
            Transaction.status.in_(['pending', 'completed', 'failed']),
            Transaction.amount != None,
            Transaction.created_at != None
        ).order_by(Transaction.created_at.desc()).limit(10).all()
        
        return render_template('admin/dashboard.html',
                             total_users=total_users,
                             total_farmers=total_farmers,
                             total_investors=total_investors,
                             total_volume=total_volume,
                             pending_users=pending_users,
                             recent_transactions=recent_transactions)
    except Exception as e:
        print(f"Admin Dashboard Error: {str(e)}")  # Log the error
        flash('An error occurred while loading the dashboard. Please try again.', 'error')
        return redirect(url_for('main.index'))

@admin_bp.route('/users')
@login_required
@admin_required
def users():
    try:
        users = User.query.order_by(User.created_at.desc()).all()
        return render_template('admin/users.html', users=users)
    except Exception as e:
        flash(f'Error loading users: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))

@admin_bp.route('/verify-user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def verify_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        user.is_verified = True
        db.session.commit()
        flash(f'User {user.name} has been verified.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error verifying user: {str(e)}', 'error')
    return redirect(request.referrer or url_for('admin.users'))

@admin_bp.route('/toggle-user-status/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def toggle_user_status(user_id):
    try:
        user = User.query.get_or_404(user_id)
        if user.role == 'admin':
            flash('Cannot modify admin user status.', 'error')
            return redirect(request.referrer or url_for('admin.users'))
            
        user.is_active = not user.is_active
        db.session.commit()
        status = 'activated' if user.is_active else 'deactivated'
        flash(f'User {user.name} has been {status}.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating user status: {str(e)}', 'error')
    return redirect(request.referrer or url_for('admin.users'))

@admin_bp.route('/transactions')
@login_required
@admin_required
def transactions():
    try:
        # Get transactions with validation
        transactions = Transaction.query.filter(
            Transaction.status.in_(['pending', 'completed', 'failed']),
            Transaction.amount != None,
            Transaction.created_at != None
        ).order_by(Transaction.created_at.desc()).all()
        
        return render_template('admin/transactions.html', transactions=transactions)
    except Exception as e:
        print(f"Admin Transactions Error: {str(e)}")  # Log the error
        flash('An error occurred while loading transactions. Please try again.', 'error')
        return redirect(url_for('admin.dashboard'))

@admin_bp.route('/soil-reports')
@login_required
@admin_required
def soil_reports():
    reports = SoilReport.query.order_by(SoilReport.report_date.desc()).all()
    return render_template('admin/soil_reports.html', reports=reports)

@admin_bp.route('/soil-report/<int:report_id>/review')
@login_required
@admin_required
def review_soil_report(report_id):
    report = SoilReport.query.get_or_404(report_id)
    if report.status != 'pending':
        flash('This report has already been reviewed.', 'warning')
        return redirect(url_for('admin.soil_reports'))
    return render_template('admin/soil_report_review.html', report=report)

@admin_bp.route('/soil-report/<int:report_id>/update', methods=['POST'])
@login_required
@admin_required
def update_soil_report(report_id):
    try:
        report = SoilReport.query.get_or_404(report_id)
        
        # Validate status
        status = request.form.get('status')
        if status not in ['approved', 'rejected']:
            flash('Invalid status value.', 'error')
            return redirect(url_for('admin.soil_reports'))
        
        # Only validate soil parameters if approving
        if status == 'approved':
            try:
                # pH Level (0-14)
                if not isinstance(report.ph_level, (int, float)) or report.ph_level < 0 or report.ph_level > 14:
                    flash('Cannot approve: pH level must be between 0 and 14.', 'error')
                    return redirect(url_for('admin.soil_reports'))
                
                # Nutrient levels (0-100%)
                for param, name in [
                    (report.nitrogen_level, 'Nitrogen'),
                    (report.phosphorus_level, 'Phosphorus'),
                    (report.potassium_level, 'Potassium'),
                    (report.organic_matter, 'Organic matter'),
                    (report.moisture_content, 'Moisture content')
                ]:
                    if not isinstance(param, (int, float)) or param < 0 or param > 100:
                        flash(f'Cannot approve: {name} level must be between 0% and 100%.', 'error')
                        return redirect(url_for('admin.soil_reports'))
                
            except (TypeError, ValueError) as e:
                flash(f'Error validating soil parameters: {str(e)}', 'error')
                return redirect(url_for('admin.soil_reports'))
        
        # Update report
        report.status = status
        report.notes = request.form.get('notes', '').strip()
        
        # Commit changes
        db.session.commit()
        
        # Flash success message
        flash(f'Soil report has been {status}.', 'success')
        
        # Log the action
        print(f"Admin {current_user.email} {status} soil report #{report.id} for farmer {report.farmer.email}")
        
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while updating the soil report: {str(e)}', 'error')
        print(f"Error updating soil report: {str(e)}")
    
    return redirect(url_for('admin.soil_reports'))

@admin_bp.route('/schemes')
@login_required
@admin_required
def schemes():
    try:
        # Get schemes with validation
        schemes = GovScheme.query.filter(
            GovScheme.start_date != None,
            GovScheme.end_date != None
        ).order_by(GovScheme.created_at.desc()).all()
        
        # Get applications with validation
        applications = SchemeApplication.query.filter(
            SchemeApplication.application_date != None
        ).order_by(SchemeApplication.application_date.desc()).all()
        
        return render_template('admin/schemes.html',
                             schemes=schemes,
                             applications=applications)
    except Exception as e:
        print(f"Admin Schemes Error: {str(e)}")  # Log the error
        flash('An error occurred while loading schemes. Please try again.', 'error')
        return redirect(url_for('admin.dashboard'))

@admin_bp.route('/scheme/add', methods=['POST'])
@login_required
@admin_required
def add_scheme():
    try:
        # Validate required fields
        name = request.form.get('name')
        description = request.form.get('description')
        eligibility = request.form.get('eligibility')
        benefits = request.form.get('benefits')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        
        if not all([name, description, eligibility, benefits, start_date_str, end_date_str]):
            flash('All fields are required.', 'error')
            return redirect(url_for('admin.schemes'))
        
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            if start_date >= end_date:
                flash('End date must be after start date.', 'error')
                return redirect(url_for('admin.schemes'))
                
            if start_date < datetime.now():
                flash('Start date cannot be in the past.', 'error')
                return redirect(url_for('admin.schemes'))
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD.', 'error')
            return redirect(url_for('admin.schemes'))
        
        scheme = GovScheme(
            name=name,
            description=description,
            eligibility_criteria=eligibility,
            benefits=benefits,
            start_date=start_date,
            end_date=end_date,
            status='active'
        )
        
        db.session.add(scheme)
        db.session.commit()
        
        flash('New government scheme added successfully!', 'success')
        return redirect(url_for('admin.schemes'))
        
    except Exception as e:
        db.session.rollback()
        print(f"Add Scheme Error: {str(e)}")  # Log the error
        flash('An error occurred while adding the scheme. Please try again.', 'error')
        return redirect(url_for('admin.schemes'))

@admin_bp.route('/scheme-application/<int:application_id>/update', methods=['POST'])
@login_required
@admin_required
def update_scheme_application(application_id):
    try:
        application = SchemeApplication.query.get_or_404(application_id)
        status = request.form.get('status')
        notes = request.form.get('notes')
        
        if not status or status not in ['approved', 'rejected']:
            flash('Invalid status specified.', 'error')
            return redirect(url_for('admin.schemes'))
        
        application.status = status
        application.notes = notes
        
        try:
            db.session.commit()
            flash(f'Scheme application has been {status}.', 'success')
        except Exception as db_error:
            db.session.rollback()
            print(f"Database Error: {str(db_error)}")
            flash('An error occurred while updating the application. Please try again.', 'error')
            
        return redirect(url_for('admin.schemes'))
        
    except Exception as e:
        print(f"Update Application Error: {str(e)}")  # Log the error
        flash('An error occurred while processing the application. Please try again.', 'error')
        return redirect(url_for('admin.schemes'))

@admin_bp.route('/chatbot')
@login_required
@admin_required
def chatbot():
    return render_template('admin/chatbot.html')

@admin_bp.route('/traceability')
@login_required
@admin_required
def traceability():
    return render_template('admin/traceability.html')

@admin_bp.route('/agri-products')
@login_required
@admin_required
def agri_products():
    # Get filter parameters
    category = request.args.get('category')
    sort = request.args.get('sort', 'price_low')
    search = request.args.get('search', '')
    
    # Base query for agricultural products
    query = Product.query.filter(
        Product.category.in_(['seeds', 'fertilizers', 'pesticides', 'equipment'])
    )
    
    # Apply filters
    if category:
        query = query.filter(Product.category == category)
    if search:
        query = query.filter(
            (Product.name.ilike(f'%{search}%')) |
            (Product.description.ilike(f'%{search}%'))
        )
    
    # Apply sorting
    if sort == 'price_low':
        query = query.order_by(Product.price.asc())
    elif sort == 'price_high':
        query = query.order_by(Product.price.desc())
    elif sort == 'name':
        query = query.order_by(Product.name.asc())
    
    products = query.all()
    return render_template('admin/agri_products.html', products=products)

@admin_bp.route('/agri-products/add', methods=['POST'])
@login_required
@admin_required
def add_agri_product():
    try:
        name = request.form.get('name')
        description = request.form.get('description')
        price = float(request.form.get('price'))
        quantity = float(request.form.get('quantity'))
        unit = request.form.get('unit')
        category = request.form.get('category')
        
        # Handle image upload
        image = request.files.get('image')
        image_url = None
        if image:
            filename = secure_filename(image.filename)
            image.save(os.path.join('app/static/uploads/products', filename))
            image_url = f'/static/uploads/products/{filename}'
        
        product = Product(
            name=name,
            description=description,
            price=price,
            quantity=quantity,
            unit=unit,
            category=category,
            image_url=image_url,
            seller_id=current_user.id,  # Admin is the seller
            status='available'
        )
        
        db.session.add(product)
        db.session.commit()
        
        flash('Agricultural product added successfully!', 'success')
        return redirect(url_for('admin.agri_products'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while adding the product: {str(e)}', 'error')
        return redirect(url_for('admin.agri_products'))

@admin_bp.route('/agri-products/<int:product_id>/edit', methods=['POST'])
@login_required
@admin_required
def edit_agri_product(product_id):
    try:
        product = Product.query.get_or_404(product_id)
        
        # Update product details
        product.name = request.form.get('name')
        product.description = request.form.get('description')
        product.price = float(request.form.get('price'))
        product.quantity = float(request.form.get('quantity'))
        product.unit = request.form.get('unit')
        product.category = request.form.get('category')
        
        # Handle image upload
        image = request.files.get('image')
        if image:
            # Delete old image if exists
            if product.image_url:
                old_image_path = os.path.join('app/static', product.image_url.lstrip('/'))
                if os.path.exists(old_image_path):
                    os.remove(old_image_path)
            
            # Save new image
            filename = secure_filename(image.filename)
            image.save(os.path.join('app/static/uploads/products', filename))
            product.image_url = f'/static/uploads/products/{filename}'
        
        db.session.commit()
        flash('Product updated successfully!', 'success')
        return redirect(url_for('admin.agri_products'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while updating the product: {str(e)}', 'error')
        return redirect(url_for('admin.agri_products'))

@admin_bp.route('/agri-products/<int:product_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_agri_product(product_id):
    try:
        product = Product.query.get_or_404(product_id)
        
        # Delete product image if exists
        if product.image_url:
            image_path = os.path.join('app/static', product.image_url.lstrip('/'))
            if os.path.exists(image_path):
                os.remove(image_path)
        
        db.session.delete(product)
        db.session.commit()
        
        flash('Product deleted successfully!', 'success')
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@admin_bp.route('/agri-products/<int:product_id>')
@login_required
@admin_required
def get_product(product_id):
    try:
        product = Product.query.get_or_404(product_id)
        return jsonify({
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': float(product.price),
            'quantity': float(product.quantity),
            'unit': product.unit,
            'category': product.category,
            'status': product.status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/orders')
@login_required
@admin_required
def orders():
    """Render the orders management page."""
    try:
        return render_template('admin/orders.html')
    except Exception as e:
        print(f"Error rendering orders page: {str(e)}")  # Debug log
        flash('An error occurred while loading the orders page. Please try again.', 'error')
        return redirect(url_for('admin.dashboard'))

@admin_bp.route('/orders/data')
@login_required
@admin_required
def orders_data():
    """Get real-time orders data."""
    try:
        # Fetch all transactions with related data
        transactions = Transaction.query.join(
            User, Transaction.buyer_id == User.id
        ).join(
            Product, Transaction.product_id == Product.id
        ).order_by(Transaction.created_at.desc()).all()

        # Separate orders by buyer role
        farmer_orders = []
        investor_orders = []
        total_orders = len(transactions)
        pending_orders = 0
        completed_orders = 0
        total_revenue = 0

        for transaction in transactions:
            order_data = {
                'id': transaction.id,
                'product': {
                    'name': transaction.product.name,
                    'category': transaction.product.category,
                    'image_url': transaction.product.image_url
                },
                'buyer': {
                    'name': transaction.buyer.name,
                    'email': transaction.buyer.email,
                    'role': transaction.buyer.role
                },
                'quantity': transaction.quantity,
                'amount': float(transaction.amount),
                'status': transaction.status,
                'created_at': transaction.created_at.isoformat(),
                'payment_method': transaction.payment_method,
                'delivery_address': transaction.delivery_address
            }

            # Update statistics
            if transaction.status == 'pending':
                pending_orders += 1
            elif transaction.status == 'completed':
                completed_orders += 1
                total_revenue += float(transaction.amount)

            # Categorize orders
            if transaction.buyer.role == 'farmer':
                farmer_orders.append(order_data)
            else:
                investor_orders.append(order_data)

        return jsonify({
            'success': True,
            'farmer_orders': farmer_orders,
            'investor_orders': investor_orders,
            'stats': {
                'total_orders': total_orders,
                'pending_orders': pending_orders,
                'completed_orders': completed_orders,
                'total_revenue': total_revenue
            }
        })
    except Exception as e:
        print(f"Error fetching orders data: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@admin_bp.route('/orders/<int:order_id>')
@login_required
@admin_required
def get_order_details(order_id):
    """Get detailed information about a specific order."""
    try:
        transaction = Transaction.query.join(
            User, Transaction.buyer_id == User.id
        ).join(
            Product, Transaction.product_id == Product.id
        ).filter(Transaction.id == order_id).first_or_404()

        return jsonify({
            'success': True,
            'id': transaction.id,
            'product': {
                'name': transaction.product.name,
                'category': transaction.product.category,
                'image_url': transaction.product.image_url
            },
            'buyer': {
                'name': transaction.buyer.name,
                'email': transaction.buyer.email,
                'role': transaction.buyer.role
            },
            'quantity': transaction.quantity,
            'amount': float(transaction.amount),
            'status': transaction.status,
            'created_at': transaction.created_at.isoformat(),
            'payment_method': transaction.payment_method,
            'delivery_address': transaction.delivery_address
        })
    except Exception as e:
        print(f"Error fetching order details: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@admin_bp.route('/orders/<int:order_id>/status', methods=['POST'])
@login_required
@admin_required
def update_order_status(order_id):
    """Update the status of an order."""
    try:
        transaction = Transaction.query.get_or_404(order_id)
        data = request.get_json()
        
        if not data or 'status' not in data:
            return jsonify({
                'success': False,
                'message': 'Status is required'
            }), 400

        new_status = data['status']
        if new_status not in ['pending', 'completed', 'failed']:
            return jsonify({
                'success': False,
                'message': 'Invalid status value'
            }), 400

        transaction.status = new_status
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Order status updated to {new_status}'
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error updating order status: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500 